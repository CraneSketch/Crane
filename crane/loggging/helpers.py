import os
import time
import json
import logging
import datetime
from collections import defaultdict, deque

import torch
import shutil


logger = logging.getLogger("crane")


class MetricLogger(object):
    def __init__(self, root_path, source_code_dir=None, task_type="Basic"):
        self.root_path = root_path
        os.makedirs(self.root_path, exist_ok=True)
        self.metirc_path = os.path.join(root_path, "train_metric.csv")
        self.final_path = os.path.join(root_path, "final_metric.csv")
        self.write_head(task_type)
        if source_code_dir is not None:
            shutil.copytree(source_code_dir, str(os.path.join(root_path, os.path.basename(source_code_dir))),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def write_head(self, task_type="Basic"):
        self.task_type = task_type
        with open(self.metirc_path, "w") as f:
            f.write("epoch,text,mae,mape,mse,mean_pred,mean_target,std_preds,std_targets,r2\n")
        with open(self.final_path, "w") as f:
            f.write("dataset,task,mape,mae,mean_pred,mean_target,activated_memory_dim\n")

    def log_metric(self, epoch, text, pred, target):
        pred = torch.cat([torch.cat(p, dim=0) for p in pred])
        target = torch.cat([torch.cat(p, dim=0) for p in target]).float()
        with torch.no_grad():
            mae = torch.mean(torch.abs(pred - target))
            nonzero = target != 0
            mape = torch.mean(torch.abs(pred[nonzero] - target[nonzero]) / target[nonzero]) if nonzero.any() else torch.tensor(float('nan'))
            mse = torch.mean(torch.square(pred - target))
            mean_pred = torch.mean(pred)
            mean_target = torch.mean(target)
            std_pred = torch.std(pred)
            std_target = torch.std(target)
            r2 = 1 - mse / torch.var(target)
            with open(self.metirc_path, "a") as f:
                f.write(f"{epoch},{text},{mae},{mape},{mse},{mean_pred},{mean_target},{std_pred},{std_target},{r2}\n")

    def log_final(self, dataset_name, task_names, preds, targets, activated_memory_dim_list):
        assert len(task_names) == len(preds) == len(targets)
        for idx in range(len(task_names)):
            with torch.no_grad():
                task_name = task_names[idx]
                pred = torch.cat(preds[idx], dim=0)
                target = torch.cat(targets[idx], dim=0).float()

                mae = torch.mean(torch.abs(pred - target))
                nonzero = target != 0
                mape = torch.mean(torch.abs(pred[nonzero] - target[nonzero]) / target[nonzero]) if nonzero.any() else torch.tensor(float('nan'))
                mean_pred = torch.mean(pred)
                mean_target = torch.mean(target)
                with open(self.final_path, "a") as f:
                    f.write(f"{dataset_name},{task_name},{mape},{mae},{mean_pred},{mean_target},{activated_memory_dim_list[idx]}\n")
