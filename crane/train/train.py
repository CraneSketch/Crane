import os
import time
import torch
import logging
from tqdm import tqdm


from omegaconf import DictConfig
from torch.utils.data import DataLoader
from crane.data import SketchDataset, MiniDataset


logger = logging.getLogger("crane")


def build_scheduler(scheduler, cfg, optimizer):
    if scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **cfg)
    elif scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cfg)
    elif scheduler == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg)
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")


def build_optimizer(optimizer, lr, model, finetune_lr=None):
    if finetune_lr is not None and hasattr(model, 'embedding_nets'):
        embedding_params = set(id(p) for p in model.embedding_nets.parameters())
        param_groups = [
            {"params": [p for p in model.embedding_nets.parameters() if p.requires_grad], "lr": finetune_lr},
            {"params": [p for n, p in model.named_parameters() if id(p) not in embedding_params and p.requires_grad], "lr": lr},
        ]
        # Remove empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        logger.info(f"Using differential learning rates: embedding_nets lr={finetune_lr}, others lr={lr}")
    else:
        param_groups = model.parameters()

    if optimizer == "AdamW":
        return torch.optim.AdamW(param_groups, lr=lr)
    elif optimizer == "Adam":
        return torch.optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def build_dataset(generator, mode=None):
    if mode == "train":
        data = generator.generate_item(generator.cfg.num_train_tasks)
        dataset = SketchDataset(data)
    elif mode == "val":
        data = generator.generate_item(generator.cfg.num_val_tasks)
        dataset = SketchDataset(data)
    elif mode == "test":
        dataset = SketchDataset(generator.cfg.real_dataset_path, device=generator.device, task_type=generator.task_type)
    else:
        raise ValueError(f"Unknown dataset builder mode: {mode}")

    return dataset


def train_one_epoch(model, train_set, optimizer, loss_fn, mini_batch_size, micro_batch_size):
    model.train()
    train_loss = 0.0
    device = next(model.parameters()).device
    preds_all = []
    targets_all = []
    max_activated_memory_dim = 0
    for sample in tqdm(train_set, desc="Training"):
        model.clear()
        optimizer.zero_grad()
        mini_batch_loss = 0.0
        preds_sample = []
        targets_sample = []

        support_x, support_y, query_x, query_y = sample
        support_x, support_y, query_x, query_y = support_x.to(device), support_y.to(device), query_x.to(device), query_y.to(device)
        support_set = MiniDataset(support_x, support_y)
        query_set = MiniDataset(query_x, query_y)
        support_mini_loader = DataLoader(support_set, batch_size=mini_batch_size, shuffle=False)
        query_mini_loader = DataLoader(query_set, batch_size=mini_batch_size, shuffle=False)

        for mini_batch in support_mini_loader:
            if mini_batch[0].shape[0] <= 1:  # Skip to avoid BatchNorm error
                continue

            mini_support_x, mini_support_y = mini_batch
            model.write(mini_support_x, mini_support_y, micro_batch_size)

        for mini_batch in query_mini_loader:
            if mini_batch[0].shape[0] <= 1:
                continue

            mini_query_x, mini_query_y = mini_batch
            preds = model.query(mini_query_x)

            loss = loss_fn(preds, mini_query_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            mini_batch_loss += loss.item()

            preds_sample.append(preds)
            targets_sample.append(mini_query_y)

        preds_all.append(preds_sample)
        targets_all.append(targets_sample)
        train_loss += mini_batch_loss / len(query_mini_loader)
        optimizer.step()
        max_activated_memory_dim = max(max_activated_memory_dim, model.activated_memory_dim)

    return train_loss / len(train_set), preds_all, targets_all, max_activated_memory_dim


def eval_one_epoch(model, eval_set, loss_fn, mini_batch_size, micro_batch_size):
    model.eval()
    eval_loss = 0.0
    device = next(model.parameters()).device
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for sample in tqdm(eval_set, desc="Evaluating"):
            model.clear()
            mini_batch_loss = 0.0
            preds_sample = []
            targets_sample = []

            support_x, support_y, query_x, query_y = sample
            support_x, support_y, query_x, query_y = support_x.to(device), support_y.to(device), query_x.to(device), query_y.to(device)
            support_set = MiniDataset(support_x, support_y)
            query_set = MiniDataset(query_x, query_y)

            support_mini_loader = DataLoader(support_set, batch_size=mini_batch_size, shuffle=False)
            query_mini_loader = DataLoader(query_set, batch_size=mini_batch_size, shuffle=False)

            for mini_batch in support_mini_loader:
                mini_support_x, mini_support_y = mini_batch
                model.write(mini_support_x, mini_support_y, micro_batch_size)

            for mini_batch in query_mini_loader:
                mini_query_x, mini_query_y = mini_batch
                preds = model.query(mini_query_x)
                mini_batch_loss += loss_fn(preds, mini_query_y)
                preds_sample.append(preds)
                targets_sample.append(mini_query_y)

            preds_all.append(preds_sample)
            targets_all.append(targets_sample)
            eval_loss += mini_batch_loss / len(query_mini_loader)

    return eval_loss / len(eval_set), preds_all, targets_all


def train(
        cfg: DictConfig,
        model,
        generator,
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
        optimizer=optimizer
    )

    best_val_loss = float("inf")
    val_set = build_dataset(generator, mode="val")
    test_set = build_dataset(generator, mode="test")
    for epoch in range(cfg.num_epoch):
        generator.refresh_base(
            regen_edges=True,
            regen_nodes=False,
            reshuffle=True,
            seed=epoch
        )
        train_set = build_dataset(generator, mode="train")

        train_loss, train_preds, train_targets, max_activated_memory_dim = train_one_epoch(
            model, train_set, optimizer, loss_fn, cfg.mini_batch_size, cfg.micro_batch_size
        )
        val_loss, val_preds, val_targets = eval_one_epoch(
            model, val_set, loss_fn, cfg.mini_batch_size, cfg.micro_batch_size
        )
        test_loss, test_preds, test_targets = eval_one_epoch(
            model, test_set, loss_fn, cfg.mini_batch_size, cfg.micro_batch_size
        )

        metric_logger.log_metric(
            epoch, "train", train_preds, train_targets
        )
        metric_logger.log_metric(
            epoch, "val", val_preds, val_targets
        )
        metric_logger.log_metric(
            epoch, "test", test_preds, test_targets
        )

        if epoch % cfg.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)

        logger.info(f"Epoch {epoch+1}/{cfg.num_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Test Loss: {test_loss}, \n"
                    f"Max Activated Memory Dim: {max_activated_memory_dim}, \n"
                    f"{model.decoder[0].weight} \n"
                    f"{model.decoder[0].bias}"
                    )

        if cfg.scheduler == "CosineAnnealingLR" or cfg.scheduler == "StepLR":
            scheduler.step()
        elif cfg.scheduler == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            raise ValueError(f"Unknown scheduler {scheduler}")

        time.sleep(0.1)
