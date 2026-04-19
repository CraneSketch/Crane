import os
import argparse

import torch
from omegaconf import OmegaConf, DictConfig

from crane.models import build_model
from crane.loggging import MetricLogger
from crane.utils import setup_logger, make_time_directories, load_merged_cfg, set_seed
from crane.eval import evaluate_nodeflow


def main(cfg: DictConfig):
    set_seed(cfg.seed)
    if not cfg.eval.dataset_path_list:
        raise ValueError("eval.dataset_path_list must not be empty")
    paths = {"out": cfg.get("out_model_path"),
             "in": cfg.get("in_model_path")}
    if not any(paths.values()):
        raise ValueError("at least one model path is required")

    log_dir = make_time_directories(os.path.join(cfg.workdir, cfg.output.log_dir),
                                    cfg.project_name, "nodeflow")
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    setup_logger(os.path.join(log_dir, "output.log"))
    metric_logger = MetricLogger(log_dir, cfg.output.source_code_dir, task_type="NodeFlow")

    models = {}
    for name, path in paths.items():
        if path is None:
            continue
        model = build_model(cfg.model).to(cfg.device)
        model.load_state_dict(torch.load(path, map_location=cfg.device))
        models[name] = model
    evaluate_nodeflow(cfg, models, metric_logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crane node-flow evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project", dest="project_name")
    parser.add_argument("--out-model-path", dest="out_model_path")
    parser.add_argument("--in-model-path", dest="in_model_path")
    args, unknown = parser.parse_known_args()
    main(load_merged_cfg(args, unknown_cli=unknown))
