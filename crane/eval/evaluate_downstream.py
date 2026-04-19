import os
import random
import logging
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from crane.data import SketchDataset, MiniDataset


logger = logging.getLogger("crane")


def decode_edge_nodes(edges_bits, node_binary_dim):
    """Decode binary-encoded edges to integer node IDs.

    Args:
        edges_bits: [N, 2 * node_binary_dim] float tensor
        node_binary_dim: number of bits per node

    Returns:
        src_ids, dst_ids: both [N] long tensors, 0-based
    """
    d = node_binary_dim
    bits = edges_bits.view(-1, 2, d)
    weights = 2 ** torch.arange(d - 1, -1, -1, device=edges_bits.device, dtype=torch.long)
    ids = torch.matmul(bits.long(), weights)  # [N, 2], 1-based
    ids = ids - 1
    return ids[:, 0], ids[:, 1]


def encode_edge_to_binary(src_id, dst_id, node_binary_dim, device="cpu"):
    """Encode a (src, dst) node pair into a binary feature vector.

    Args:
        src_id: integer node ID (0-based)
        dst_id: integer node ID (0-based)
        node_binary_dim: number of bits per node

    Returns:
        [2 * node_binary_dim] float tensor
    """
    d = node_binary_dim
    shifts = torch.arange(d - 1, -1, -1, device=device, dtype=torch.long)
    src_val = src_id + 1  # 1-based for encoding
    dst_val = dst_id + 1
    src_bits = ((src_val >> shifts) & 1).float()
    dst_bits = ((dst_val >> shifts) & 1).float()
    return torch.cat([src_bits, dst_bits], dim=0)


def encode_edges_batch(src_ids, dst_ids, node_binary_dim, device="cpu"):
    """Batch-encode multiple (src, dst) pairs into binary feature vectors.

    Args:
        src_ids: list/tensor of 0-based source node IDs
        dst_ids: list/tensor of 0-based dest node IDs
        node_binary_dim: bits per node

    Returns:
        [N, 2 * node_binary_dim] float tensor
    """
    d = node_binary_dim
    shifts = torch.arange(d - 1, -1, -1, device=device, dtype=torch.long)
    src_vals = torch.tensor(src_ids, device=device, dtype=torch.long) + 1
    dst_vals = torch.tensor(dst_ids, device=device, dtype=torch.long) + 1
    src_bits = ((src_vals.unsqueeze(1) >> shifts) & 1).float()
    dst_bits = ((dst_vals.unsqueeze(1) >> shifts) & 1).float()
    return torch.cat([src_bits, dst_bits], dim=1)


def build_adjacency_from_support(support_x, support_y, node_binary_dim):
    """Build directed adjacency and weight dict from support data.

    Args:
        support_x: [N, 2*d] edge features
        support_y: [N] edge weights
        node_binary_dim: bits per node

    Returns:
        adjacency: dict {src_id: [dst_id, ...]}
        weight_dict: dict {(src_id, dst_id): total_weight}
        edge_set: set of (src_id, dst_id) tuples
    """
    src_ids, dst_ids = decode_edge_nodes(support_x, node_binary_dim)
    src_ids_cpu = src_ids.cpu()
    dst_ids_cpu = dst_ids.cpu()
    weights_cpu = support_y.cpu().float()

    # Vectorized aggregation: pack (src, dst) into single key via Cantor-like encoding
    max_id = max(src_ids_cpu.max().item(), dst_ids_cpu.max().item()) + 1
    edge_keys = src_ids_cpu.long() * max_id + dst_ids_cpu.long()

    unique_keys, inverse = torch.unique(edge_keys, return_inverse=True)
    agg_weights = torch.zeros(unique_keys.shape[0], dtype=torch.float32)
    agg_weights.scatter_add_(0, inverse, weights_cpu)

    unique_src = (unique_keys // max_id).tolist()
    unique_dst = (unique_keys % max_id).tolist()
    agg_w = agg_weights.tolist()

    adjacency = defaultdict(list)
    weight_dict = {}
    for s, d, w in zip(unique_src, unique_dst, agg_w):
        adjacency[s].append(d)
        weight_dict[(s, d)] = w

    edge_set = set(weight_dict.keys())
    return dict(adjacency), dict(weight_dict), edge_set


def generate_path_queries(adjacency, weight_dict, node_binary_dim,
                          num_queries, min_path_length, max_path_length,
                          negative_ratio, seed, device="cpu"):
    """Generate path queries via random walks on the adjacency graph.

    Returns:
        query_edges_list: list of [K, 2*d] tensors (edges in path)
        query_targets: [num_queries] tensor of ground truth max-flow values
    """
    rng = random.Random(seed)
    nodes_with_out = [n for n, nbrs in adjacency.items() if len(nbrs) > 0]

    num_positive = max(1, int(num_queries * (1.0 - negative_ratio)))
    num_negative = num_queries - num_positive

    paths = []
    gt_flows = []
    max_attempts = num_positive * 3

    # Positive paths: random walks where all edges exist
    attempts = 0
    while len(paths) < num_positive and attempts < max_attempts:
        attempts += 1
        path_len = rng.randint(min_path_length, max_path_length)
        start = rng.choice(nodes_with_out)
        path_nodes = [start]
        valid = True
        for _ in range(path_len):
            cur = path_nodes[-1]
            if cur not in adjacency or len(adjacency[cur]) == 0:
                valid = False
                break
            nxt = rng.choice(adjacency[cur])
            path_nodes.append(nxt)
        if not valid or len(path_nodes) < min_path_length + 1:
            continue

        src_list = [path_nodes[i] for i in range(len(path_nodes) - 1)]
        dst_list = [path_nodes[i + 1] for i in range(len(path_nodes) - 1)]
        path_tensor = encode_edges_batch(src_list, dst_list, node_binary_dim, device=device)
        min_weight = min(weight_dict[(s, d)] for s, d in zip(src_list, dst_list))

        paths.append(path_tensor)
        gt_flows.append(min_weight)

    if len(paths) < num_positive:
        logger.warning(f"Only generated {len(paths)}/{num_positive} positive path queries "
                       f"(graph may have limited connectivity)")

    # Negative paths: take a positive path, replace one edge with a non-existent one.
    # Paper's path-flow definition: f_p(psi) = 0 if any edge in psi has f(e)=0,
    # else min(f(e_i)). A non-existent edge has true weight 0, so gt_flow = 0.
    all_nodes = list(set(k for pair in weight_dict for k in pair))
    edge_set = set(weight_dict.keys())
    neg_generated = 0
    neg_attempts = 0
    max_neg_attempts = num_negative * 5
    while neg_generated < num_negative and neg_attempts < max_neg_attempts and len(paths) > 0:
        neg_attempts += 1
        base_idx = rng.randint(0, len(gt_flows) - 1)
        base_path = paths[base_idx]
        path_len = base_path.shape[0]
        replace_idx = rng.randint(0, path_len - 1)

        for _ in range(20):
            s = rng.choice(all_nodes)
            d = rng.choice(all_nodes)
            if (s, d) not in edge_set:
                new_path = base_path.clone()
                new_path[replace_idx] = encode_edges_batch([s], [d], node_binary_dim, device=device).squeeze(0)
                paths.append(new_path)
                gt_flows.append(0.0)
                neg_generated += 1
                break

    if neg_generated < num_negative:
        logger.warning(f"Only generated {neg_generated}/{num_negative} negative path queries")

    query_targets = torch.tensor(gt_flows, dtype=torch.float32, device=device)
    return paths, query_targets


def generate_subgraph_queries(weight_dict, edge_set, node_binary_dim,
                              num_queries, min_subgraph_size, max_subgraph_size,
                              nonexistent_ratio, seed, device="cpu"):
    """Generate subgraph queries by sampling edge subsets.

    Returns:
        query_edges_list: list of [M, 2*d] tensors (edges in subgraph)
        query_targets: [num_queries] tensor of ground truth total weights
    """
    rng = random.Random(seed)
    edge_list = list(weight_dict.keys())
    all_nodes = list(set(k for pair in edge_list for k in pair))

    queries = []
    gt_weights = []

    for _ in range(num_queries):
        subgraph_size = rng.randint(min_subgraph_size, max_subgraph_size)
        num_nonexistent = max(0, int(subgraph_size * nonexistent_ratio))
        num_existing = subgraph_size - num_nonexistent

        # Sample existing edges
        num_existing = min(num_existing, len(edge_list))
        sampled_existing = rng.sample(edge_list, num_existing)

        # Sample non-existent edges
        nonexistent_edges = []
        ne_attempts = 0
        while len(nonexistent_edges) < num_nonexistent and ne_attempts < num_nonexistent * 20:
            ne_attempts += 1
            s = rng.choice(all_nodes)
            d = rng.choice(all_nodes)
            if (s, d) not in edge_set:
                nonexistent_edges.append((s, d))

        all_edges = sampled_existing + nonexistent_edges
        rng.shuffle(all_edges)

        src_list = [s for s, d in all_edges]
        dst_list = [d for s, d in all_edges]
        edge_tensor = encode_edges_batch(src_list, dst_list, node_binary_dim, device=device)
        total_weight = sum(weight_dict.get((s, d), 0.0) for s, d in all_edges)

        queries.append(edge_tensor)
        gt_weights.append(total_weight)

    query_targets = torch.tensor(gt_weights, dtype=torch.float32, device=device)
    return queries, query_targets


def path_query(qw_model, path_edges):
    """Execute a Path Flow Query.

    Paper Eq. 3: f_p(psi) = 0 if any edge has f(e)=0 else min(f(e_i)).
    Non-existent edges are expected to yield near-zero Qw estimates, so the
    min reduction naturally implements the paper's definition using Qw alone.

    Returns:
        scalar tensor: predicted max flow
    """
    weights = qw_model.query(path_edges)
    return weights.min()


def subgraph_query(qw_model, subgraph_edges):
    """Execute a Subgraph Flow Query.

    Paper Eq. 4: f_g(G_Q) = sum_{e in E_Q} f(e). Non-existent edges contribute
    zero weight by definition, so the sum over Qw estimates matches the paper.

    Returns:
        scalar tensor: predicted total weight of the subgraph
    """
    weights = qw_model.query(subgraph_edges)
    return weights.sum()


def _load_downstream_queries(task_dir, device):
    """Load pre-stored path/subgraph queries from downstream.npz."""
    npz_path = os.path.join(task_dir, "downstream.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Pre-stored downstream queries not found: {npz_path}. "
            f"Run: python -m crane.data.generate_downstream_splits "
            f"--dataset-root <dataset_root>"
        )

    with np.load(npz_path, allow_pickle=False) as data:
        path_edges = torch.tensor(data["path_edges"], dtype=torch.float32, device=device)
        path_offsets = data["path_offsets"]
        path_targets = torch.tensor(data["path_targets"], dtype=torch.float32, device=device)
        sg_edges = torch.tensor(data["sg_edges"], dtype=torch.float32, device=device)
        sg_offsets = data["sg_offsets"]
        sg_targets = torch.tensor(data["sg_targets"], dtype=torch.float32, device=device)

    path_queries = [path_edges[path_offsets[i]:path_offsets[i + 1]]
                    for i in range(len(path_offsets) - 1)]
    sg_queries = [sg_edges[sg_offsets[i]:sg_offsets[i + 1]]
                  for i in range(len(sg_offsets) - 1)]
    return path_queries, path_targets, sg_queries, sg_targets


def _scan_and_sort_task_dirs(root_path):
    dir_list = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    try:
        dir_list = sorted(dir_list, key=lambda s: int(os.path.basename(s).split("_")[1]))
    except (IndexError, ValueError):
        dir_list = sorted(dir_list)
    return dir_list


def eval_downstream_one_task(qw_model, eval_set, dataset_root, cfg):
    """Evaluate downstream tasks (Path Flow + Subgraph Flow) on one dataset.

    Loads pre-stored path/subgraph queries from <task_dir>/downstream.npz.

    Returns:
        path_preds, path_targets, subgraph_preds, subgraph_targets,
        activated_memory_dim_list
    """
    qw_model.eval()

    device = next(qw_model.parameters()).device
    mini_batch_size = cfg.eval.mini_batch_size
    micro_batch_size = cfg.eval.micro_batch_size

    task_dirs = _scan_and_sort_task_dirs(dataset_root)

    path_preds_all = []
    path_targets_all = []
    subgraph_preds_all = []
    subgraph_targets_all = []
    activated_memory_dim_list = []

    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm(eval_set, desc=f"Downstream eval {eval_set.name}")):
            qw_model.clear()

            support_x, support_y, _, _ = sample

            support_set = MiniDataset(support_x, support_y)
            support_loader = DataLoader(support_set, batch_size=mini_batch_size, shuffle=False)

            for mini_batch in support_loader:
                mini_x, mini_y = mini_batch
                mini_x = mini_x.to(device)
                mini_y = mini_y.to(device)
                qw_model.write(mini_x, mini_y, micro_batch_size)

            task_dir = os.path.join(dataset_root, task_dirs[sample_idx])
            path_queries, path_gt, sg_queries, sg_gt = _load_downstream_queries(task_dir, device)

            path_preds_sample = []
            for pq_edges in path_queries:
                pred = path_query(qw_model, pq_edges)
                path_preds_sample.append(pred)

            if path_preds_sample:
                path_preds_all.append(torch.stack(path_preds_sample))
                path_targets_all.append(path_gt)

            sg_preds_sample = []
            for sg_edges in sg_queries:
                pred = subgraph_query(qw_model, sg_edges)
                sg_preds_sample.append(pred)

            if sg_preds_sample:
                subgraph_preds_all.append(torch.stack(sg_preds_sample))
                subgraph_targets_all.append(sg_gt)

            activated_memory_dim_list.append(qw_model.activated_memory_dim)
            torch.cuda.empty_cache()

    return (path_preds_all, path_targets_all,
            subgraph_preds_all, subgraph_targets_all,
            activated_memory_dim_list)


def evaluate_downstream(cfg, qw_model, metric_logger):
    """Run downstream evaluation (Path Flow + Subgraph Flow) on all datasets.

    Requires pre-stored downstream.npz in each task dir. Generate via:
        python -m crane.data.generate_downstream_splits --dataset-root <root>
    """
    qw_model = qw_model.to(cfg.eval.model_device)

    for task_path in cfg.eval.dataset_path_list:
        dataset = SketchDataset(task_path, "Basic", cfg.eval.data_device, lazy=True)
        logger.info(f"Downstream evaluation on {dataset.name}")

        (path_preds, path_targets,
         sg_preds, sg_targets,
         memory_dims) = eval_downstream_one_task(qw_model, dataset, task_path, cfg)

        # Log Path Flow Query metrics
        if path_preds:
            path_preds_wrapped = [[p] for p in path_preds]
            path_targets_wrapped = [[t] for t in path_targets]
            path_task_names = [f"task_{i}" for i in range(len(path_preds))]
            metric_logger.log_final(
                f"{dataset.name}_PathQuery",
                path_task_names,
                path_preds_wrapped,
                path_targets_wrapped,
                memory_dims,
            )

        # Log Subgraph Flow Query metrics
        if sg_preds:
            sg_preds_wrapped = [[p] for p in sg_preds]
            sg_targets_wrapped = [[t] for t in sg_targets]
            sg_task_names = [f"task_{i}" for i in range(len(sg_preds))]
            metric_logger.log_final(
                f"{dataset.name}_SubGraphQuery",
                sg_task_names,
                sg_preds_wrapped,
                sg_targets_wrapped,
                memory_dims,
            )
