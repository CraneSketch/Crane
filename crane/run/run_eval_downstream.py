import os
import argparse

import torch
from omegaconf import OmegaConf, DictConfig

from crane.models import build_model
from crane.loggging import MetricLogger
from crane.utils import setup_logger, make_time_directories, load_merged_cfg, set_seed
from crane.eval import evaluate_downstream


def main(cfg: DictConfig):
    if hasattr(cfg, 'seed'):
        set_seed(cfg.seed)

    qw_model = build_model(cfg.qw_model).to(cfg.device)

    log_dir = make_time_directories(os.path.join(cfg.workdir, cfg.output.log_dir), cfg.project_name, "downstream")
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    setup_logger(os.path.join(log_dir, "output.log"))

    metric_logger = MetricLogger(log_dir, source_code_dir=cfg.output.source_code_dir, task_type="Basic")

    qw_model.load_state_dict(torch.load(cfg.qw_model_path, map_location=cfg.device))

    evaluate_downstream(cfg, qw_model, metric_logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crane downstream evaluation")
    parser.add_argument("--config", type=str, help="path to yaml config")
    parser.add_argument("--project", type=str, dest="project_name")
    parser.add_argument("--qw-model-path", type=str, dest="qw_model_path", required=True,
                        help="path to trained Crane (Qw) checkpoint")
    args, unknown = parser.parse_known_args()

    config = load_merged_cfg(args, unknown_cli=unknown)
    main(config)
