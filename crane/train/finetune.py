import os
import time
import torch
import logging

from omegaconf import DictConfig
from crane.train.train import build_optimizer, build_scheduler, train_one_epoch, eval_one_epoch


logger = logging.getLogger("crane")


def finetune(
        cfg: DictConfig,
        model,
        train_dataset,
        val_dataset,
        loss_fn,
        metric_logger,
        checkpoint_dir,
        finetune_lr=None,
):
    optimizer = build_optimizer(
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        model=model,
        finetune_lr=finetune_lr,
    )
    scheduler = build_scheduler(
        scheduler=cfg.scheduler,
        cfg=cfg.scheduler_config,
        optimizer=optimizer,
    )

    best_val_loss = float("inf")

    for epoch in range(cfg.num_epoch):
        train_loss, train_preds, train_targets, max_activated_memory_dim = train_one_epoch(
            model, train_dataset, optimizer, loss_fn, cfg.mini_batch_size, cfg.micro_batch_size
        )
        val_loss, val_preds, val_targets = eval_one_epoch(
            model, val_dataset, loss_fn, cfg.mini_batch_size, cfg.micro_batch_size
        )

        metric_logger.log_metric(epoch, "finetune_train", train_preds, train_targets)
        metric_logger.log_metric(epoch, "finetune_val", val_preds, val_targets)

        if epoch % cfg.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)

        logger.info(
            f"Finetune Epoch {epoch+1}/{cfg.num_epoch}, "
            f"Train Loss: {train_loss}, Val Loss: {val_loss}, "
            f"Max Activated Memory Dim: {max_activated_memory_dim}"
        )

        if cfg.scheduler == "CosineAnnealingLR" or cfg.scheduler == "StepLR":
            scheduler.step()
        elif cfg.scheduler == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            raise ValueError(f"Unknown scheduler {cfg.scheduler}")

        time.sleep(0.1)
