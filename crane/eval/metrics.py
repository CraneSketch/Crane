import torch


def are(pred, target):
    mask = target != 0
    if not mask.any():
        return torch.tensor(float("nan"))
    return torch.mean(torch.abs(pred[mask] - target[mask]) / target[mask])


def recall_at_k(pred, target, k):
    if target.numel() == 0 or k <= 0:
        return torch.tensor(float("nan"))
    k = min(int(k), target.numel())
    return torch.isin(torch.topk(pred, k).indices,
                      torch.topk(target, k).indices).float().mean()


def ndcg_at_k(pred, target, k):
    if target.numel() == 0 or k <= 0:
        return torch.tensor(float("nan"))
    k = min(int(k), target.numel())
    discount = 1 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32,
                                            device=target.device))
    ranked = target[torch.topk(pred, k).indices]
    ideal = torch.topk(target, k).values
    denominator = torch.sum(ideal * discount)
    return torch.sum(ranked * discount) / denominator if denominator > 0 else torch.tensor(float("nan"))


def build_metric_row(task_name, pred, target, activated_memory_dim, topk_list=(100,), seed=""):
    pred = pred.detach().float().cpu().flatten()
    target = target.detach().float().cpu().flatten()
    if pred.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} vs {target.shape}")
    positive = target != 0
    ranked_pred, ranked_target = pred[positive], target[positive]
    row = {
        "task": task_name,
        "num_queries": target.numel(),
        "num_zero_gt": int((target == 0).sum()),
        "are": are(pred, target),
        "aae": torch.mean(torch.abs(pred - target)),
        "mean_pred": torch.mean(pred),
        "mean_target": torch.mean(target),
    }
    for k in topk_list:
        row[f"recall_at_{int(k)}"] = recall_at_k(ranked_pred, ranked_target, k)
        row[f"ndcg_at_{int(k)}"] = ndcg_at_k(ranked_pred, ranked_target, k)
    row["activated_memory_dim"] = activated_memory_dim
    row["seed"] = seed
    return row
