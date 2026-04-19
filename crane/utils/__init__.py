import os
import random
import logging
import argparse
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

import torch


def setup_logger(
    log_path,
    level=logging.INFO,
    to_console=True,
):
    logger = logging.getLogger("crane")
    logger.setLevel(level)
    logger.propagate = False         # Prevent propagation to root to avoid duplicate output

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_console:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    fh = TimedRotatingFileHandler(
        log_path, when="D", interval=1, backupCount=7, encoding="utf-8"
    )

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def make_time_directories(base_dir=".", project_name="default", run_type="train", now=None):
    """Create project/run-type timestamped directories for logging output.

    Creates: {base_dir}/{project_name}/{run_type}_{YYYY-MM-DD}_{HH-MM-SS}/
    """
    if now is None:
        now = datetime.now()

    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, project_name, f"{timestamp}_{run_type}")

    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def load_merged_cfg(args: argparse.Namespace, unknown_cli=None) -> DictConfig:
    base = OmegaConf.load(args.config) if getattr(args, "config", None) else OmegaConf.create({})

    cli_known = OmegaConf.create({
        k: v for k, v in vars(args).items()
        if k != "config" and v is not None
    })

    dotlist = [tok for tok in (unknown_cli or []) if "=" in tok and not tok.startswith("--")]
    cli_dot = OmegaConf.from_dotlist(dotlist) if dotlist else OmegaConf.create({})

    merged = OmegaConf.merge(base, cli_known, cli_dot)
    return merged


def set_seed(seed: int):
    """Set random seeds for all random number generators to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure cuDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
