import os
import argparse
import logging

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from crane.models import build_model
from crane.data import SketchDataset
from crane.train import finetune
from crane.loss import build_loss_fn
from crane.loggging import MetricLogger
from crane.utils import setup_logger, make_time_directories, load_merged_cfg, set_seed
from crane.eval import eval_one_task


logger = logging.getLogger("crane")


def _scan_and_sort_task_dirs(root_path):
    dir_list = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    try:
        dir_list = sorted(dir_list, key=lambda s: int(os.path.basename(s).split("_")[1]))
    except (IndexError, ValueError):
        dir_list = sorted(dir_list)
    return dir_list


def _load_single_npz(npz_path, task_type, device):
    with np.load(npz_path, allow_pickle=False) as data:
        support_x = torch.tensor(data["support_x"], dtype=torch.float32, device=device)
        support_y = torch.tensor(data["support_y"], dtype=torch.long, device=device)

        if task_type == "Basic":
            query_x = torch.tensor(data["query_edge_x"], dtype=torch.float32, device=device)
            query_y = torch.tensor(data["query_edge_y"], dtype=torch.long, device=device).squeeze(-1)
        elif task_type == "Degree":
            query_x = torch.tensor(data["query_node_x"], dtype=torch.float32, device=device)
            query_y = torch.tensor(data["query_node_y"], dtype=torch.long, device=device).squeeze(-1)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    return (support_x, support_y, query_x, query_y)


def main(cfg: DictConfig):
    if hasattr(cfg, 'seed'):
        set_seed(cfg.seed)

    log_dir = make_time_directories(os.path.join(cfg.workdir, cfg.output.log_dir), cfg.project_name, "finetune")
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    setup_logger(os.path.join(log_dir, "output.log"))
    metric_logger = MetricLogger(log_dir, cfg.output.source_code_dir, task_type=cfg.task_type)

    is_finetune_dir = os.path.isdir(cfg.load_model_path)
    if is_finetune_dir:
        pretrained_state = None
        logger.info(f"load_model_path is a directory; loading per-task from "
                    f"{cfg.load_model_path}/checkpoint/<dataset>/<task>/best_model.pth")
    else:
        pretrained_state = torch.load(cfg.load_model_path, map_location=cfg.device)
    loss_fn = build_loss_fn(cfg.loss_fn)

    for dataset_path in cfg.eval.dataset_path_list:
        dataset_name = os.path.basename(dataset_path)
        subdirs = _scan_and_sort_task_dirs(dataset_path)

        for subdir in subdirs:
            subdir_path = os.path.join(dataset_path, subdir)
            train_npz = os.path.join(subdir_path, "finetune_train.npz")
            test_npz = os.path.join(subdir_path, "finetune_test.npz")
            eval_npz = os.path.join(subdir_path, "0.npz")

            if not os.path.isfile(train_npz) or not os.path.isfile(test_npz):
                logger.warning(f"Skipping {subdir_path}: missing finetune npz files")
                continue

            logger.info(f"=== Finetuning {dataset_name}/{subdir} ===")

            if is_finetune_dir:
                per_task_ckpt = os.path.join(cfg.load_model_path, "checkpoint",
                                             dataset_name, subdir, "best_model.pth")
                if not os.path.isfile(per_task_ckpt):
                    logger.warning(f"Skipping {subdir_path}: missing per-task ckpt {per_task_ckpt}")
                    continue
                task_pretrained_state = torch.load(per_task_ckpt, map_location=cfg.device)
                load_source = per_task_ckpt
            else:
                task_pretrained_state = pretrained_state
                load_source = cfg.load_model_path

            model = build_model(cfg.model).to(cfg.device)
            strict = cfg.finetune.strict
            if not strict:
                model_state = model.state_dict()
                filtered = {k: v for k, v in task_pretrained_state.items()
                            if k in model_state and v.shape == model_state[k].shape}
                skipped = set(task_pretrained_state.keys()) - set(filtered.keys())
                if skipped:
                    logger.info(f"Skipped loading keys (shape mismatch or missing): {skipped}")
                model.load_state_dict(filtered, strict=False)
            else:
                model.load_state_dict(task_pretrained_state, strict=True)
            logger.info(f"Loaded pretrained weights from {load_source} (strict={strict})")

            if cfg.finetune.freeze_embedding_nets:
                for param in model.embedding_nets.parameters():
                    param.requires_grad = False

            train_item = _load_single_npz(train_npz, cfg.task_type, cfg.device)
            val_item = _load_single_npz(test_npz, cfg.task_type, cfg.device)
            train_dataset = SketchDataset([train_item])
            val_dataset = SketchDataset([val_item])

            checkpoint_dir = os.path.join(log_dir, "checkpoint", dataset_name, subdir)
            os.makedirs(checkpoint_dir, exist_ok=True)

            finetune(
                cfg.train,
                model,
                train_dataset,
                val_dataset,
                loss_fn,
                metric_logger,
                checkpoint_dir,
                finetune_lr=cfg.finetune.finetune_lr,
            )

            model.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "best_model.pth"), map_location=cfg.eval.model_device)
            )
            model = model.to(cfg.eval.model_device)

            eval_item = _load_single_npz(eval_npz, cfg.task_type, cfg.eval.data_device)
            eval_dataset = SketchDataset([eval_item], task_type=cfg.task_type, device=cfg.eval.data_device)
            eval_dataset.name = dataset_name
            eval_dataset.task_names = [subdir]

            preds, targets, activated_memory_dim_list = eval_one_task(
                model, eval_dataset, cfg.eval.mini_batch_size, cfg.eval.micro_batch_size
            )
            metric_logger.log_final(dataset_name, [subdir], preds, targets, activated_memory_dim_list)

            logger.info(f"=== Done {dataset_name}/{subdir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crane finetuning on real data")
    parser.add_argument("--config", type=str, help="path to yaml config")
    parser.add_argument("--workdir", type=str, dest="working_dictionary")
    parser.add_argument("--project", type=str, dest="project_name")
    parser.add_argument("--load-model-path", type=str, dest="load_model_path", required=True)
    args, unknown = parser.parse_known_args()

    config = load_merged_cfg(args, unknown_cli=unknown)
    main(config)
