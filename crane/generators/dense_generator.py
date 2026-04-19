import torch
from tqdm import tqdm

class DenseGenerator:
    def __init__(
            self,
            cfg,
            task_type,
            node_binary_dim,
            item_lower,
            item_upper,
            generate_ratio,
            ave_frequency,
            zipf_param_lower,
            zipf_param_upper,
            skew_lower,
            skew_upper
    ):
        self.cfg = cfg
        self.task_type = task_type
        self.device = torch.device("cpu")              # Output device; used only when returning results
        self._base_device = torch.device("cpu")        # Fixed device for generation and sampling (CPU)

        self.node_binary_dim = int(node_binary_dim)
        self.item_lower = int(item_lower)
        self.item_upper = int(item_upper)
        self.generate_ratio = int(generate_ratio)
        self.ave_frequency = float(ave_frequency)
        self.zipf_param_lower = float(zipf_param_lower)
        self.zipf_param_upper = float(zipf_param_upper)
        self.skew_lower = int(skew_lower)
        self.skew_upper = int(skew_upper)
        self.max_nodes = 1_000_000

        self._test_zip = None

        self._nodes = self._generate_nodes_cpu()
        self._edges, self._order, self._ptr = self._generate_edges_cpu(), None, 0
        self._reshuffle_cpu()

    def set_device(self, device):
        self.device = torch.device(device)

    def _generate_nodes_cpu(self):
        max_possible = 2 ** self.node_binary_dim
        num_nodes = min(self.max_nodes, max_possible - 1)
        idx = torch.arange(1, num_nodes + 1, device=self._base_device, dtype=torch.long)
        shifts = torch.arange(self.node_binary_dim - 1, -1, -1, device=self._base_device, dtype=torch.long)
        bits = ((idx.unsqueeze(1) >> shifts) & 1).to(torch.float32)
        return bits

    def _generate_edges_cpu(self):
        n = self._nodes.shape[0]
        m = max(n * self.generate_ratio, n)
        src = torch.randint(0, n, (m,), device=self._base_device)
        dst = torch.randint(0, n, (m,), device=self._base_device)
        edges = torch.cat([self._nodes[src], self._nodes[dst]], dim=1)
        edges = torch.unique(edges, dim=0)
        return edges.to(torch.float32)

    def _reshuffle_cpu(self):
        self._order = torch.randperm(self._edges.shape[0], device=self._base_device)
        self._ptr = 0

    def _ensure_capacity_cpu(self, item_size):
        if item_size > self._edges.shape[0]:
            self._edges = self._generate_edges_cpu()
            self._reshuffle_cpu()
        if self._ptr + item_size > self._order.shape[0]:
            self._reshuffle_cpu()

    def _sample_items_cpu(self, item_size):
        self._ensure_capacity_cpu(item_size)
        idx = self._order[self._ptr:self._ptr + item_size]
        self._ptr += item_size
        return self._edges[idx]

    def _base_freq_cpu(self, n):
        return torch.ones(n, device=self._base_device, dtype=torch.float32) * self.ave_frequency

    def _apply_skew_cpu(self, y, skew_ratio):
        if skew_ratio >= 1:
            return torch.round(y * skew_ratio)
        else:
            return torch.round(y * skew_ratio) + 1

    def _zipf_counts_cpu(self, zipf_param, size, stream_length):
        x = torch.arange(1, size + 1, device=self._base_device, dtype=torch.float32)
        w = x.pow(-zipf_param)
        w = w / w.sum()
        c = torch.round(w * stream_length)
        p = torch.randperm(size, device=self._base_device)
        return c[p] + 1

    def set_once_zip_param(self, param):
        self._test_zip = float(param)

    def _decode_edge_nodes_cpu(self, edges_bits: torch.Tensor):
        """Decode edge binary encoding back to node IDs (0-based).

        Args:
            edges_bits: [m, 2 * node_binary_dim], concatenation of [src_bits, dst_bits]

        Returns:
            src_ids, dst_ids: both [m] long tensors, 0-based node indices
        """
        d = self.node_binary_dim
        bits = edges_bits.view(-1, 2, d).to(self._base_device)  # [m, 2, d]
        weights = (2 ** torch.arange(d - 1, -1, -1, device=self._base_device, dtype=torch.long))  # [d]
        ids = torch.matmul(bits.long(), weights)  # [m, 2], range [1, 2^d - 1]
        ids = ids - 1  # Map to 0-based
        src_ids = ids[:, 0].long()
        dst_ids = ids[:, 1].long()
        return src_ids, dst_ids

    def sample_one_support(self, item_size=None, skew_ratio=None, zipf_param=None, shuffle_items=True):
        if item_size is None:
            item_size = int(torch.randint(self.item_lower, self.item_upper, (1,), device=self._base_device).item())

        items = self._sample_items_cpu(item_size)

        # Step 1: Apply skew distribution across samples
        y = self._base_freq_cpu(item_size)
        if skew_ratio is None:
            skew_ratio = int(torch.randint(self.skew_lower, self.skew_upper, (1,), device=self._base_device).item())
        y = self._apply_skew_cpu(y, skew_ratio)

        # Step 2: Apply Zipf distribution to edge frequencies
        if zipf_param is None:
            zipf_param = (self.zipf_param_upper - self.zipf_param_lower) * torch.rand(1, device=self._base_device).item() + self.zipf_param_lower
        y = self._zipf_counts_cpu(float(zipf_param), item_size, int(y.sum().item()))

        # Step 3: Shuffle order
        if shuffle_items:
            p = torch.randperm(item_size, device=self._base_device)
            items = items[p]
            y = y[p]

        return items.to(self.device), y.to(self.device)

    def generate_item(self, num_tasks, item_size=None):
        data = []
        for _ in tqdm(range(int(num_tasks)), desc="Generating items..."):
            support_x, support_y = self.sample_one_support(item_size=item_size)
            # Copy to _base_device for CPU-based statistics/sampling
            support_x_cpu = support_x.to(self._base_device)
            support_y_cpu = support_y.to(self._base_device)

            if self.task_type == "Basic":
                # Basic task: edge-level regression, use same support set as query
                query_x = support_x
                query_y = support_y

            elif self.task_type == "Degree":
                # Degree task: compute cumulative in/out flow for each node
                # 1. Decode src/dst node IDs from support_x
                src_ids, dst_ids = self._decode_edge_nodes_cpu(support_x_cpu)  # [m], [m]
                n_nodes = self._nodes.shape[0]

                # 2. Accumulate flow using scatter_add (support_y as weights)
                in_flow = torch.zeros(n_nodes, device=self._base_device, dtype=torch.float32)
                out_flow = torch.zeros_like(in_flow)

                in_flow.scatter_add_(0, dst_ids, support_y_cpu)
                out_flow.scatter_add_(0, src_ids, support_y_cpu)

                # 3. Keep only nodes that appeared in this task
                mask = (in_flow > 0) | (out_flow > 0)
                node_ids = mask.nonzero(as_tuple=False).squeeze(1)  # [k]

                # Use binary encoding of corresponding nodes as query_x
                query_x_cpu = self._nodes[node_ids]  # [k, node_binary_dim]

                # query_y is cumulative in+out flow for each node
                query_y_cpu = in_flow[node_ids] + out_flow[node_ids]

                # 4. Shuffle node order to avoid ordering bias
                if node_ids.numel() > 1:
                    p = torch.randperm(node_ids.numel(), device=self._base_device)
                    query_x_cpu = query_x_cpu[p]
                    query_y_cpu = query_y_cpu[p]

                query_x = query_x_cpu.to(self.device)
                query_y = query_y_cpu.to(self.device)

            else:
                raise NotImplementedError(f"Unknown task_type: {self.task_type}")

            data.append((support_x, support_y, query_x, query_y))

        return data

    def refresh_base(self, regen_edges=True, regen_nodes=False, reshuffle=True, seed=None):
        if seed is not None:
            torch.manual_seed(int(seed))

        if regen_nodes:
            self._nodes = self._generate_nodes_cpu()
            self._edges = self._generate_edges_cpu()
            if reshuffle:
                self._reshuffle_cpu()
            else:
                self._ptr = 0
            return

        if regen_edges:
            self._edges = self._generate_edges_cpu()

        if reshuffle:
            self._reshuffle_cpu()
        else:
            self._ptr = 0
