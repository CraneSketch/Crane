import os
import argparse

import torch
from omegaconf import OmegaConf, DictConfig


from crane.models import build_model
from crane.loggging import MetricLogger
from crane.utils import setup_logger, make_time_directories, load_merged_cfg, set_seed
from crane.eval import evaluate_on_real_data


def main(cfg: DictConfig):
    # Set random seed for reproducibility
    if hasattr(cfg, 'seed'):
        set_seed(cfg.seed)

    model = build_model(cfg.model).to(cfg.device)
    log_dir = make_time_directories(os.path.join(cfg.workdir, cfg.output.log_dir), cfg.project_name, "eval")
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    setup_logger(
        os.path.join(log_dir, "output.log")
    )
    metric_logger = MetricLogger(log_dir, source_code_dir=cfg.output.source_code_dir, task_type=cfg.task_type)
    load_model_path = cfg.load_model_path
    model.load_state_dict(torch.load(load_model_path, map_location=cfg.eval.model_device))
    evaluate_on_real_data(
        cfg.eval,
        model,
        cfg.task_type,
        metric_logger,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crane runner")
    parser.add_argument("--config", type=str, help="path to yaml config")
    parser.add_argument("--project", type=str, dest="project_name")
    parser.add_argument("--load-model-path", type=str, dest="load_model_path")
    args, unknown = parser.parse_known_args()

    config = load_merged_cfg(args, unknown_cli=unknown)
    main(config)
