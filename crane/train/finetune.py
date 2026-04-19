import os
import time
import torch
import logging

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from crane.data import MiniDataset
from crane.train.train import build_optimizer, build_scheduler, train_one_epoch, eval_one_epoch


logger = logging.getLogger("crane")


def _cache_tasks(model, dataset, mini_batch_size, micro_batch_size):
    device = next(model.parameters()).device
    cached = []
    model.eval()
    with torch.no_grad():
        for support_x, support_y, query_x, query_y in dataset:
            model.clear()
            loader = DataLoader(MiniDataset(support_x, support_y),
                                batch_size=mini_batch_size, shuffle=False)
            for mini_x, mini_y in loader:
                model.write(mini_x.to(device).float(), mini_y.to(device), micro_batch_size)
            cached.append((model.memory_matrix.clone(), model.activated_memory_dim,
                           query_x.to(device), query_y.to(device)))
    return cached


def _cached_epoch(model, cached, loss_fn, mini_batch_size, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        model.embedding_nets.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []
    max_dim = 0
    for memory, active_dim, query_x, query_y in cached:
        model.memory_matrix.copy_(memory)
        model.activated_memory_dim = active_dim
        loader = DataLoader(MiniDataset(query_x, query_y), batch_size=mini_batch_size,
                            shuffle=training)
        preds_sample, targets_sample = [], []
        for mini_x, mini_y in loader:
            if training and len(mini_x) <= 1:
                continue
            if training:
                optimizer.zero_grad()
            with torch.set_grad_enabled(training):
                pred = model.query(mini_x.float())
                loss = loss_fn(pred, mini_y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
            total_loss += loss.item()
            preds_sample.append(pred.detach())
            targets_sample.append(mini_y)
        preds_all.append(preds_sample)
        targets_all.append(targets_sample)
        max_dim = max(max_dim, active_dim)
    return total_loss / max(sum(len(p) for p in preds_all), 1), preds_all, targets_all, max_dim


def finetune(
        cfg: DictConfig,
        model,
        train_dataset,
        val_dataset,
        loss_fn,
        metric_logger,
        checkpoint_dir,
        finetune_lr=None,
        cache_written_memory=False,
        select_by="val_loss",
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

    use_cache = (cache_written_memory and hasattr(model, "embedding_nets")
                 and all(not p.requires_grad for p in model.embedding_nets.parameters()))
    if use_cache:
        train_cache = _cache_tasks(model, train_dataset, cfg.mini_batch_size,
                                   cfg.micro_batch_size)
        val_cache = _cache_tasks(model, val_dataset, cfg.mini_batch_size,
                                 cfg.micro_batch_size)

    best_val_loss = float("inf")

    for epoch in range(cfg.num_epoch):
        if use_cache:
            train_loss, train_preds, train_targets, max_activated_memory_dim = _cached_epoch(
                model, train_cache, loss_fn, cfg.mini_batch_size, optimizer)
            val_loss, val_preds, val_targets, _ = _cached_epoch(
                model, val_cache, loss_fn, cfg.mini_batch_size)
        else:
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

        val_value = float(val_loss)
        if select_by == "last" or val_value < best_val_loss:
            best_val_loss = val_value
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
