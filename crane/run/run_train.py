import os
import argparse
import logging

import torch
from omegaconf import OmegaConf, DictConfig


from crane.models import build_model
from crane.generators import build_generator
from crane.train import train
from crane.loss import build_loss_fn
from crane.loggging import MetricLogger
from crane.utils import setup_logger, make_time_directories, load_merged_cfg, set_seed
from crane.eval import evaluate_on_real_data


logger = logging.getLogger("crane")


def main(cfg: DictConfig):
    # Set random seed for reproducibility
    if hasattr(cfg, 'seed'):
        set_seed(cfg.seed)

    model = build_model(cfg.model).to(cfg.device)
    generator = build_generator(cfg.generator, task_type=cfg.task_type)
    loss_fn = build_loss_fn(cfg.loss_fn)
    log_dir = make_time_directories(os.path.join(cfg.workdir, cfg.output.log_dir), cfg.project_name, "train")
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    setup_logger(
        os.path.join(log_dir, "output.log")
    )
    metric_logger = MetricLogger(log_dir, cfg.output.source_code_dir, task_type=cfg.task_type)

    # Load pretrained weights if specified
    if hasattr(cfg, 'load_model_path') and cfg.load_model_path is not None:
        strict = cfg.finetune.strict
        state_dict = torch.load(cfg.load_model_path, map_location=cfg.device)
        if not strict:
            model_state = model.state_dict()
            filtered = {k: v for k, v in state_dict.items()
                        if k in model_state and v.shape == model_state[k].shape}
            skipped = set(state_dict.keys()) - set(filtered.keys())
            if skipped:
                logger.info(f"Skipped loading keys (shape mismatch or missing): {skipped}")
            model.load_state_dict(filtered, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded pretrained weights from {cfg.load_model_path} (strict={strict})")

    # Optionally freeze embedding_nets
    if hasattr(cfg, 'finetune') and cfg.finetune.freeze_embedding_nets:
        for param in model.embedding_nets.parameters():
            param.requires_grad = False
        logger.info("Froze embedding_nets parameters")

    finetune_lr = None
    if hasattr(cfg, 'finetune') and cfg.finetune.finetune_lr is not None:
        finetune_lr = cfg.finetune.finetune_lr
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    train(
        cfg.train,
        model,
        generator,
        loss_fn,
        metric_logger,
        checkpoint_dir,
        finetune_lr=finetune_lr,
    )
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_model.pth"), map_location=cfg.eval.model_device))
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
