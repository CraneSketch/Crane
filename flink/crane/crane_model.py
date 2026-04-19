import torch
import torch.nn as nn
import torch.nn.functional as F


def _fold_bn(linear, bn):
    """Fold BatchNorm1d into preceding Linear (exact in eval mode).

    BN(Linear(x)) = scale * (Wx + b) + bias_bn = (scale*W)x + (scale*b + bias_bn)
    """
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    W_fused = linear.weight * scale.unsqueeze(1)
    b_fused = scale * linear.bias + bn.bias - scale * bn.running_mean
    return W_fused.data, b_fused.data


class Crane(nn.Module):
    def __init__(
            self,
            source_input_dim: int = 32,
            dest_input_dim: int = 32,
            source_hidden_dim: int = 32,
            dest_hidden_dim: int = 32,
            source_embedding_dim: int = 64,
            dest_embedding_dim: int = 64,
            memory_layer: int = 4,
            carry_threshold: int = 4,
            use_fused_kernels: bool = False,
    ):
        super(Crane, self).__init__()
        self.memory_layer = memory_layer
        self.carry_threshold = carry_threshold
        self.activated_memory_dim = None
        self.use_fused_kernels = use_fused_kernels
        self._optimized = False

        self.embedding_nets = nn.ModuleList()
        for _ in range(memory_layer):
            source_net = nn.Sequential(
                nn.Linear(source_input_dim, source_hidden_dim),
                nn.BatchNorm1d(source_hidden_dim),
                nn.ReLU(),
                nn.Linear(source_hidden_dim, (source_hidden_dim + source_embedding_dim) // 2),
                nn.BatchNorm1d((source_hidden_dim + source_embedding_dim) // 2),
                nn.ReLU(),
                nn.Linear((source_hidden_dim + source_embedding_dim) // 2, source_embedding_dim),
                nn.Softplus(),
            )
            dest_net = nn.Sequential(
                nn.Linear(dest_input_dim, dest_hidden_dim),
                nn.BatchNorm1d(dest_hidden_dim),
                nn.ReLU(),
                nn.Linear(dest_hidden_dim, (dest_hidden_dim + dest_embedding_dim) // 2),
                nn.BatchNorm1d((dest_hidden_dim + dest_embedding_dim) // 2),
                nn.ReLU(),
                nn.Linear((dest_hidden_dim + dest_embedding_dim) // 2, dest_embedding_dim),
                nn.Softplus(),
            )
            self.embedding_nets.append(nn.ModuleList([source_net, dest_net]))

        self.decoder = nn.Sequential(
            nn.Linear(memory_layer, 1),
        )
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.register_buffer(
            "memory_matrix",
            torch.zeros(memory_layer, source_embedding_dim, dest_embedding_dim)
        )

    @torch.no_grad()
    def optimize_for_inference(self, use_fp16=True):
        """Fold BN into Linear, batch all 8 embedding nets, optionally FP16.

        Call after eval() and to(device). BN folding is mathematically exact.
        """
        L = self.memory_layer
        device = self.memory_matrix.device
        dtype = torch.float16 if (use_fp16 and device.type == 'cuda') else torch.float32

        # Fold BN into Linear for all 8 sub-networks (4 source + 4 dest)
        # Interleaved order: src_0, dst_0, src_1, dst_1, ...
        w1_all, b1_all = [], []
        w2_all, b2_all = [], []
        w3_all, b3_all = [], []

        for k in range(L):
            for net_idx in (0, 1):  # 0=source, 1=dest
                net = self.embedding_nets[k][net_idx]
                # net = [Linear, BN, ReLU, Linear, BN, ReLU, Linear, Softplus]
                w1, b1 = _fold_bn(net[0], net[1])  # Layer 1
                w2, b2 = _fold_bn(net[3], net[4])  # Layer 2
                w3, b3 = net[6].weight.data, net[6].bias.data  # Layer 3 (no BN)
                w1_all.append(w1); b1_all.append(b1)
                w2_all.append(w2); b2_all.append(b2)
                w3_all.append(w3); b3_all.append(b3)

        src_idx = list(range(0, 2 * L, 2))  # source net indices
        dst_idx = list(range(1, 2 * L, 2))  # dest net indices

        # Layer 1: concatenated weights [in, L*out] for single matmul
        self.register_buffer('_w1_src',
            torch.cat([w1_all[i].t() for i in src_idx], dim=1).to(dtype=dtype, device=device))
        self.register_buffer('_b1_src',
            torch.stack([b1_all[i] for i in src_idx]).to(dtype=dtype, device=device))
        self.register_buffer('_w1_dst',
            torch.cat([w1_all[i].t() for i in dst_idx], dim=1).to(dtype=dtype, device=device))
        self.register_buffer('_b1_dst',
            torch.stack([b1_all[i] for i in dst_idx]).to(dtype=dtype, device=device))

        # Layers 2-3: BMM weights [L, in, out]
        self.register_buffer('_w2_src',
            torch.stack([w2_all[i].t() for i in src_idx]).to(dtype=dtype, device=device))
        self.register_buffer('_b2_src',
            torch.stack([b2_all[i] for i in src_idx]).unsqueeze(1).to(dtype=dtype, device=device))
        self.register_buffer('_w2_dst',
            torch.stack([w2_all[i].t() for i in dst_idx]).to(dtype=dtype, device=device))
        self.register_buffer('_b2_dst',
            torch.stack([b2_all[i] for i in dst_idx]).unsqueeze(1).to(dtype=dtype, device=device))

        self.register_buffer('_w3_src',
            torch.stack([w3_all[i].t() for i in src_idx]).to(dtype=dtype, device=device))
        self.register_buffer('_b3_src',
            torch.stack([b3_all[i] for i in src_idx]).unsqueeze(1).to(dtype=dtype, device=device))
        self.register_buffer('_w3_dst',
            torch.stack([w3_all[i].t() for i in dst_idx]).to(dtype=dtype, device=device))
        self.register_buffer('_b3_dst',
            torch.stack([b3_all[i] for i in dst_idx]).unsqueeze(1).to(dtype=dtype, device=device))

        # Pre-allocate write buffers to avoid per-call allocation
        L_dim, R_dim, C_dim = self.memory_matrix.shape
        self.register_buffer('_base_buf',
            torch.empty(L_dim, R_dim, C_dim, device=device, dtype=torch.float32))
        self.register_buffer('_base_sum_buf',
            torch.empty(L_dim, R_dim, C_dim, device=device, dtype=torch.float32))

        self._optimized = True
        self._embed_dtype = dtype

        # torch.compile on embedding — disabled for now (autotuning overhead
        # exceeds benefit in short-lived Flink jobs)
        # if device.type == 'cuda':
        #     self._get_embedding_inner = torch.compile(
        #         self._get_embedding_inner, mode="max-autotune-no-cudagraphs")

    def get_embedding(self, input_x: torch.Tensor):
        if not self._optimized:
            return self._get_embedding_original(input_x)

        half = input_x.shape[1] // 2
        dt = self._embed_dtype
        source_x = input_x[:, :half].to(dt)
        dest_x = input_x[:, half:].to(dt)
        return self._get_embedding_inner(source_x, dest_x)

    def _embedding_layers(self, source_x, dest_x):
        """Run all 3 embedding layers, return [L, B, emb] in native dtype."""
        B = source_x.shape[0]
        L = self.memory_layer

        # Layer 1: [B,32] @ [32, L*h1] → [B, L*h1] → reshape → [L, B, h1]
        src = torch.relu(
            (source_x @ self._w1_src).view(B, L, -1).permute(1, 0, 2).contiguous()
            + self._b1_src.unsqueeze(1)
        )
        dst = torch.relu(
            (dest_x @ self._w1_dst).view(B, L, -1).permute(1, 0, 2).contiguous()
            + self._b1_dst.unsqueeze(1)
        )

        # Layer 2: BMM [L,B,h1] @ [L,h1,h2] → [L, B, h2]
        src = torch.relu(torch.bmm(src, self._w2_src) + self._b2_src)
        dst = torch.relu(torch.bmm(dst, self._w2_dst) + self._b2_dst)

        # Layer 3: BMM [L,B,h2] @ [L,h2,emb] → [L, B, emb]
        src = F.softplus(torch.bmm(src, self._w3_src) + self._b3_src)
        dst = F.softplus(torch.bmm(dst, self._w3_dst) + self._b3_dst)
        return src, dst  # [L, B, emb] in native dtype (FP16 on CUDA)

    def _get_embedding_inner(self, source_x, dest_x):
        src, dst = self._embedding_layers(source_x, dest_x)
        # [L, B, emb] → [B, L, emb] in FP32 for BLR kernels / training
        source_emb = src.permute(1, 0, 2).contiguous().float()
        dest_emb = dst.permute(1, 0, 2).contiguous().float()
        return source_emb, dest_emb

    def get_embedding_lbr(self, input_x: torch.Tensor):
        """Return embeddings in [L, B, emb] layout without permute/copy.

        For use with fused_query_min_inference_lbr kernel only.
        """
        if not self._optimized:
            # Fallback: compute via original path and permute back
            src, dst = self._get_embedding_original(input_x)
            return src.permute(1, 0, 2).contiguous(), dst.permute(1, 0, 2).contiguous()

        half = input_x.shape[1] // 2
        dt = self._embed_dtype
        source_x = input_x[:, :half].to(dt)
        dest_x = input_x[:, half:].to(dt)
        return self._embedding_layers(source_x, dest_x)

    def _get_embedding_original(self, input_x: torch.Tensor):
        B, D = input_x.shape[0], input_x.shape[1]
        half = D // 2
        source_x = input_x[:, :half]
        dest_x = input_x[:, half:]

        source_list, dest_list = [], []
        for k in range(self.memory_layer):
            src_net, dst_net = self.embedding_nets[k]
            source_list.append(src_net(source_x))
            dest_list.append(dst_net(dest_x))

        source_emb = torch.stack(source_list, dim=1)
        dest_emb = torch.stack(dest_list, dim=1)
        return source_emb, dest_emb

    @torch.no_grad()
    def clear(self):
        self.memory_matrix.zero_()
        self.activated_memory_dim = 0

    @torch.no_grad()
    def write(self, input_x: torch.Tensor, input_y: torch.Tensor, micro_batch_size: int = 4):
        source_emb, dest_emb = self.get_embedding(input_x)  # [B, L, R], [B, L, C]
        B, L, R = source_emb.shape
        _, Ld, C = dest_emb.shape
        assert L == self.memory_layer and Ld == L

        if self.use_fused_kernels and source_emb.is_cuda:
            from crane.crane_kernels import fused_write_sum_triton, fused_carry
            input_y_flat = input_y.view(B)
            base_buf = self._base_buf
            base_sum_buf = self._base_sum_buf
            for i in range(0, B, micro_batch_size):
                mb = min(micro_batch_size, B - i)
                s = source_emb[i:i + mb]
                d = dest_emb[i:i + mb]
                y = input_y_flat[i:i + mb]
                fused_write_sum_triton(s, d, y, L, R, C, base_buf, base_sum_buf)
                fused_carry(self.memory_matrix, base_buf, base_sum_buf, self.carry_threshold)
            self.activated_memory_dim = L - 1
        else:
            input_y = input_y.view(B, 1, 1, 1).expand(B, self.memory_layer, 1, 1)
            base_all = torch.einsum('blr,blc->blrc', source_emb, dest_emb)
            size_all = input_y
            dtype = self.memory_matrix[0].dtype
            base_all = base_all.to(dtype)
            L = self.memory_layer
            for i in range(0, B, micro_batch_size):
                base_mini_batch = base_all[i: i + micro_batch_size]
                size_mini_batch = size_all[i: i + micro_batch_size]
                base_per_layer = base_mini_batch.sum(dim=0)
                base_sum_per_layer = (base_mini_batch * size_mini_batch).sum(dim=0)
                self.memory_matrix[0].add_(base_sum_per_layer[0])
                for j in range(L - 1):
                    nz = base_per_layer[j] != 0
                    need_carry = self.memory_matrix[j][nz] / (self.carry_threshold * base_per_layer[j][nz])
                    if need_carry.min() >= 1:
                        carry_number = need_carry.min()
                        self.memory_matrix[j].sub_(self.carry_threshold * carry_number * base_per_layer[j])
                        self.memory_matrix[j + 1].add_(base_per_layer[j + 1] * carry_number)
                        self.activated_memory_dim = max(self.activated_memory_dim, j + 1)
                    if j + 1 > self.activated_memory_dim:
                        break

    def query(self, input_x: torch.Tensor) -> torch.Tensor:
        source_emb, dest_emb = self.get_embedding(input_x)   # [B, L, R], [B, L, C]
        B, L, R = source_emb.shape
        _, Ld, C = dest_emb.shape
        assert Ld == L == self.memory_layer

        if self.use_fused_kernels and source_emb.is_cuda:
            if self.training:
                from crane.crane_kernels import fused_query_min
                ratio_min = fused_query_min(source_emb, dest_emb, self.memory_matrix, self.activated_memory_dim)
            else:
                from crane.crane_kernels.query_kernel import fused_query_min_inference
                ratio_min = fused_query_min_inference(source_emb, dest_emb, self.memory_matrix, self.activated_memory_dim)
        else:
            base_per_layer = torch.einsum('blr,blc->blrc', source_emb, dest_emb)
            memory_now = self.memory_matrix.unsqueeze(0).expand(B, -1, -1, -1)
            nz = base_per_layer != 0
            ratio_map = torch.zeros_like(base_per_layer)
            ratio_map[nz] = memory_now[nz] / base_per_layer[nz]
            inf = torch.full_like(base_per_layer, float('inf'))
            masked = torch.where(nz, ratio_map, inf)
            layer_min, _ = masked.view(B, L, -1).min(dim=-1)
            has_valid = nz.view(B, L, -1).any(dim=-1)
            ratio_min = torch.zeros(B, L, device=base_per_layer.device, dtype=base_per_layer.dtype)
            ratio_min[has_valid] = layer_min[has_valid]
            if self.activated_memory_dim < (L - 1):
                mask = torch.zeros(L, dtype=torch.bool, device=ratio_min.device)
                mask[: self.activated_memory_dim + 1] = True
                ratio_min = ratio_min * mask.unsqueeze(0)

        return self.decoder(ratio_min).squeeze(-1)           # [B, 1]

    def query_inference(self, input_x: torch.Tensor, ratio_min_buf=None) -> torch.Tensor:
        """Fast inference-only query using LBR-layout kernel.

        Skips the expensive permute+contiguous+float() by keeping embeddings
        in [L, B, emb] layout and native dtype (FP16 on CUDA).
        """
        src_lbr, dst_lbr = self.get_embedding_lbr(input_x)  # [L, B, R], [L, B, C]
        L, B, R = src_lbr.shape
        C = dst_lbr.shape[2]
        assert L == self.memory_layer

        if self.use_fused_kernels and src_lbr.is_cuda:
            from crane.crane_kernels.query_kernel import fused_query_min_inference_lbr
            ratio_min = fused_query_min_inference_lbr(
                src_lbr, dst_lbr, self.memory_matrix,
                self.activated_memory_dim, out=ratio_min_buf)
        else:
            # CPU fallback: convert to BLR for the einsum path
            source_emb = src_lbr.permute(1, 0, 2).contiguous().float()
            dest_emb = dst_lbr.permute(1, 0, 2).contiguous().float()
            base_per_layer = torch.einsum('blr,blc->blrc', source_emb, dest_emb)
            memory_now = self.memory_matrix.unsqueeze(0).expand(B, -1, -1, -1)
            nz = base_per_layer != 0
            ratio_map = torch.zeros_like(base_per_layer)
            ratio_map[nz] = memory_now[nz] / base_per_layer[nz]
            inf = torch.full_like(base_per_layer, float('inf'))
            masked = torch.where(nz, ratio_map, inf)
            layer_min, _ = masked.view(B, L, -1).min(dim=-1)
            has_valid = nz.view(B, L, -1).any(dim=-1)
            ratio_min = torch.zeros(B, L, device=base_per_layer.device, dtype=base_per_layer.dtype)
            ratio_min[has_valid] = layer_min[has_valid]
            if self.activated_memory_dim < (L - 1):
                mask = torch.zeros(L, dtype=torch.bool, device=ratio_min.device)
                mask[: self.activated_memory_dim + 1] = True
                ratio_min = ratio_min * mask.unsqueeze(0)

        return self.decoder(ratio_min).squeeze(-1)
