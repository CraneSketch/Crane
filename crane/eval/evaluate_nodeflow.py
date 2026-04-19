import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from crane.data import SketchDataset, MiniDataset
from crane.eval.metrics import build_metric_row
from crane.utils import weight_scale_for


def _query(model, query_x, batch_size):
    parts = [model.query(query_x[i:i + batch_size].float())
             for i in range(0, len(query_x), batch_size)]
    return torch.cat(parts) if parts else torch.empty(0, device=query_x.device)


def evaluate_nodeflow(cfg, models, metric_logger):
    models = {name: model.to(cfg.eval.model_device)
              for name, model in models.items() if model is not None}
    if not models:
        raise ValueError("at least one node-flow model is required")
    for model in models.values():
        model.eval()

    device = next(next(iter(models.values())).parameters()).device
    topk_list = list(cfg.nodeflow.get("topk_list", [100]))

    for dataset_path in cfg.eval.dataset_path_list:
        dataset = SketchDataset(dataset_path, "Node", cfg.eval.data_device, lazy=True)
        scale = weight_scale_for(cfg, dataset.name)
        rows = {name: [] for name in models}

        with torch.no_grad():
            for index, sample in enumerate(dataset):
                support_x, support_y, _, _ = sample
                for model in models.values():
                    model.clear()
                loader = DataLoader(MiniDataset(support_x, support_y),
                                    batch_size=cfg.eval.mini_batch_size, shuffle=False)
                for mini_x, mini_y in loader:
                    mini_x = mini_x.to(device).float()
                    mini_y = mini_y.to(device) * scale
                    for model in models.values():
                        model.write(mini_x, mini_y, cfg.eval.micro_batch_size)

                task_name = dataset.task_names[index]
                task_dir = os.path.join(dataset_path, task_name)
                queries = {}
                with np.load(os.path.join(task_dir, "nodeflow.npz"), allow_pickle=False) as data:
                    node_x = torch.as_tensor(data["node_x"], device=device)
                    queries["out"] = (node_x, torch.as_tensor(data["out_y"], dtype=torch.float32,
                                                               device=device).reshape(-1) * scale)
                    queries["in"] = (node_x, torch.as_tensor(data["in_y"], dtype=torch.float32,
                                                              device=device).reshape(-1) * scale)

                for name, model in models.items():
                    query_x, target = queries[name]
                    pred = _query(model, query_x, cfg.eval.mini_batch_size)
                    rows[name].append(build_metric_row(task_name, pred, target,
                                                       model.activated_memory_dim, topk_list,
                                                       cfg.get("seed", "")))
                dataset.release_cache()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        suffixes = {"out": "NodeFlowOut", "in": "NodeFlowIn"}
        for name, result in rows.items():
            metric_logger.log_downstream_final(f"{dataset.name}_{suffixes[name]}", result)
