import torch
import torch.nn as nn


class CraneForPathFlow(nn.Module):
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
            id_decode: bool = True,
            id_decode_hidden: int = 512,
            id_embed_buckets: int = 262144,
            id_embed_dim: int = 4,
    ):
        super(CraneForPathFlow, self).__init__()
        self.memory_layer = memory_layer
        self.carry_threshold = carry_threshold
        self.activated_memory_dim = None
        self.id_decode = id_decode
        self.id_embed_buckets = id_embed_buckets

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

        if id_decode:
            decode_dim = memory_layer + source_input_dim + dest_input_dim
            if id_embed_buckets > 0:
                if id_embed_buckets < 4:
                    raise ValueError("id_embed_buckets must be at least 4")
                self.id_embed_pair = nn.Embedding(id_embed_buckets, id_embed_dim)
                self.id_embed_src = nn.Embedding(id_embed_buckets // 4, id_embed_dim)
                self.id_embed_dst = nn.Embedding(id_embed_buckets // 4, id_embed_dim)
                for table in (self.id_embed_pair, self.id_embed_src, self.id_embed_dst):
                    nn.init.normal_(table.weight, std=0.05)
                decode_dim += 3 * id_embed_dim
                self.register_buffer("_bit_weights",
                                     2 ** torch.arange(source_input_dim - 1, -1, -1,
                                                       dtype=torch.long), persistent=False)
            self.decoder = nn.Sequential(
                nn.Linear(decode_dim, id_decode_hidden), nn.ReLU(),
                nn.Linear(id_decode_hidden, id_decode_hidden), nn.ReLU(),
                nn.Linear(id_decode_hidden, 1),
            )
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)
        else:
            self.decoder = nn.Sequential(nn.Linear(memory_layer, 1))
            nn.init.zeros_(self.decoder[0].weight)
            nn.init.zeros_(self.decoder[0].bias)

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
                need_carry = self.memory_matrix[j] / (self.carry_threshold * base_per_layer[j])

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

        base_per_layer = torch.einsum('blr,blc->blrc', source_emb, dest_emb)  # [B, L, R, C]
        memory_now = self.memory_matrix.unsqueeze(0).expand(B, -1, -1, -1)    # [B, L, R, C]
        ratio_map = torch.zeros_like(base_per_layer)
        ratio_map = memory_now / base_per_layer
        inf = torch.full_like(base_per_layer, float('inf'))
        layer_min, _ = ratio_map.view(B, L, -1).min(dim=-1)    # [B, L]

        if self.activated_memory_dim < (L - 1):
            mask = torch.zeros(L, dtype=torch.bool, device=layer_min.device)
            mask[: self.activated_memory_dim + 1] = True
            layer_min = layer_min * mask.unsqueeze(0)        # [B, L]
        if self.id_decode:
            parts = [torch.sign(layer_min) * torch.log1p(layer_min.abs()), input_x.float()]
            if self.id_embed_buckets > 0:
                half = input_x.shape[1] // 2
                src_id = (input_x[:, :half].long() * self._bit_weights).sum(-1)
                dst_id = (input_x[:, half:].long() * self._bit_weights).sum(-1)
                pair_idx = ((src_id * 2654435761 + dst_id) * 2246822519).remainder(self.id_embed_buckets)
                src_idx = (src_id * 2654435761).remainder(self.id_embed_buckets // 4)
                dst_idx = (dst_id * 2246822519).remainder(self.id_embed_buckets // 4)
                parts.extend((self.id_embed_pair(pair_idx), self.id_embed_src(src_idx),
                              self.id_embed_dst(dst_idx)))
            return torch.expm1(torch.clamp(self.decoder(torch.cat(parts, dim=-1)).squeeze(-1), max=30.0))
        return self.decoder(layer_min).squeeze(-1)           # [B, 1]

    def query_path(self, path_edges: torch.Tensor) -> torch.Tensor:
        return self.query(path_edges).min()
