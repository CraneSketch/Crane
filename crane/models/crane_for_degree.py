import torch
import torch.nn as nn


class CraneForDegree(nn.Module):
    def __init__(
            self,
            source_input_dim: int,
            dest_input_dim: int,
            source_hidden_dim: int,
            dest_hidden_dim: int,
            source_embedding_dim: int,
            dest_embedding_dim: int,
            memory_layer: int,
            carry_threshold: int,
    ):
        super(CraneForDegree, self).__init__()
        self.memory_layer = memory_layer
        self.carry_threshold = carry_threshold
        self.activated_memory_dim = None

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

    def get_embedding(self, input_x: torch.Tensor):
        B, D = input_x.shape[0], input_x.shape[1]
        half = D // 2
        source_x = input_x[:, :half]
        dest_x = input_x[:, half:]

        source_list, dest_list = [], []
        for k in range(self.memory_layer):
            src_net, dst_net = self.embedding_nets[k]
            source_list.append(src_net(source_x))  # [B, R]
            dest_list.append(dst_net(dest_x))      # [B, C]

        source_emb = torch.stack(source_list, dim=1)  # [B, L, R]
        dest_emb = torch.stack(dest_list, dim=1)      # [B, L, C]
        return source_emb, dest_emb

    @torch.no_grad()
    def clear(self):
        self.memory_matrix.zero_()
        self.activated_memory_dim = 0

    @torch.no_grad()
    def write(self, input_x: torch.Tensor, input_y: torch.Tensor, mini_batch_size: int = 4):
        source_emb, dest_emb = self.get_embedding(input_x)  # [B, L, R], [B, L, C]
        B, L, R = source_emb.shape
        _, Ld, C = dest_emb.shape
        assert L == self.memory_layer and Ld == L

        input_y = input_y.view(B, 1, 1, 1).expand(B, self.memory_layer, 1, 1)
        base_all = torch.einsum('blr,blc->blrc', source_emb, dest_emb)
        size_all = input_y
        dtype = self.memory_matrix[0].dtype
        base_all = base_all.to(dtype)

        L = self.memory_layer

        # Process data in mini-batches
        for i in range(0, B, mini_batch_size):
            base_mini_batch = base_all[i: i + mini_batch_size]
            size_mini_batch = size_all[i: i + mini_batch_size]
            base_per_layer = base_mini_batch.sum(dim=0)  # [L, R, C]
            base_sum_per_layer = (base_mini_batch * size_mini_batch).sum(dim=0)
            self.memory_matrix[0].add_(base_sum_per_layer[0])
            for j in range(L - 1):  # excluding the top layer
                nz = base_per_layer[j] != 0
                need_carry = self.memory_matrix[j][nz] / (self.carry_threshold * base_per_layer[j][nz])

                if need_carry.min() >= 1:
                    carry_number = need_carry.min()
                    self.memory_matrix[j].sub_(self.carry_threshold * carry_number * base_per_layer[j])
                    self.memory_matrix[j + 1].add_(base_per_layer[j + 1] * carry_number)
                    self.activated_memory_dim = max(self.activated_memory_dim, j + 1)

                if j + 1 > self.activated_memory_dim:
                    break

    def query(self, input_node: torch.Tensor) -> torch.Tensor:
        input_x = torch.cat((input_node, input_node), dim=-1)

        source_emb, dest_emb = self.get_embedding(input_x)   # [B, L, R], [B, L, C]
        B, L, R = source_emb.shape
        _, Ld, C = dest_emb.shape
        assert Ld == L == self.memory_layer

        base_per_layer = torch.einsum('blr,blc->blrc', source_emb, dest_emb)  # [B, L, R, C]
        memory_now = self.memory_matrix.unsqueeze(0).expand(B, -1, -1, -1)    # [B, L, R, C]
        nz = base_per_layer != 0
        ratio_map = torch.zeros_like(base_per_layer)
        ratio_map[nz] = memory_now[nz] / base_per_layer[nz]
        inf = torch.full_like(base_per_layer, float('inf'))
        masked = torch.where(nz, ratio_map, inf)            # [B, L, R, C]
        layer_min, _ = masked.view(B, L, -1).min(dim=-1)    # [B, L]
        has_valid = nz.view(B, L, -1).any(dim=-1)           # [B, L]
        ratio_min = torch.zeros(B, L, device=base_per_layer.device, dtype=base_per_layer.dtype)
        ratio_min[has_valid] = layer_min[has_valid]

        if self.activated_memory_dim < (L - 1):
            mask = torch.zeros(L, dtype=torch.bool, device=ratio_min.device)
            mask[: self.activated_memory_dim + 1] = True
            ratio_min = ratio_min * mask.unsqueeze(0)        # [B, L]
        return self.decoder(ratio_min).squeeze(-1)           # [B, 1]