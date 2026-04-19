import os
import gc
import numpy as np
import torch
import logging

from torch.utils.data import Dataset
from typing import List, Tuple, Optional


logger = logging.getLogger("crane")


class FinetuneDataset(Dataset):
    """Load pre-split finetune data from finetune_train.npz / finetune_test.npz.

    split="train" loads 10% queries for fine-tuning.
    split="test"  loads 90% queries for evaluation.
    task_type decides which query fields to read (Basic -> query_edge, Degree -> query_node).
    """

    _SPLIT_FILE = {
        "train": "finetune_train.npz",
        "test": "finetune_test.npz",
    }

    def __init__(self, root_path: str, task_type: str, split: str = "train",
                 device: Optional[torch.device | str] = None, lazy: bool = False):
        super().__init__()
        self.name = os.path.basename(root_path)
        self.task_type = task_type
        self.device = device
        self.lazy = lazy
        self.split = split

        self._npz_paths: Optional[List[str]] = None
        self._cached_idx: Optional[int] = None
        self._cached_item: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None

        filename = self._SPLIT_FILE[split]
        task_names, dir_list = self._scan_and_sort_task_dirs(root_path)
        self.task_names = []

        if lazy:
            npz_paths = []
            for folder in dir_list:
                npz_path = os.path.join(root_path, folder, filename)
                if not os.path.isfile(npz_path):
                    logger.warning(f"npz not found: {npz_path}")
                    continue
                npz_paths.append(npz_path)
                self.task_names.append(folder)
            self._npz_paths = npz_paths
            self.data = None
        else:
            self.data = []
            for folder in dir_list:
                npz_path = os.path.join(root_path, folder, filename)
                if not os.path.isfile(npz_path):
                    logger.warning(f"npz not found: {npz_path}")
                    continue
                item = self._load_one_npz(npz_path, device_override=device)
                self.data.append(item)
                self.task_names.append(folder)

        if self.device is None:
            self.device = torch.device("cpu")

    def __getitem__(self, index: int):
        if self.lazy:
            if self._cached_idx == index and self._cached_item is not None:
                return self._cached_item
            item = self._load_one_npz(self._npz_paths[index])
            self._cached_idx = index
            self._cached_item = item
            return item
        return self.data[index]

    def __len__(self) -> int:
        if self.lazy:
            return len(self._npz_paths)
        return len(self.data)

    def release_cache(self):
        self._cached_idx = None
        self._cached_item = None
        gc.collect()

    def _scan_and_sort_task_dirs(self, root_path: str):
        dir_list = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        try:
            dir_list = sorted(dir_list, key=lambda s: int(os.path.basename(s).split("_")[1]))
        except (IndexError, ValueError):
            dir_list = sorted(dir_list)
        return dir_list[:], dir_list

    def _load_one_npz(self, npz_path: str, device_override=None):
        dev = device_override if device_override is not None else self.device
        with np.load(npz_path, allow_pickle=False) as data:
            support_x = torch.tensor(data["support_x"], dtype=torch.float32, device=dev)
            support_y = torch.tensor(data["support_y"], dtype=torch.long, device=dev)

            if self.task_type == "Basic":
                query_x = torch.tensor(data["query_edge_x"], dtype=torch.float32, device=dev)
                query_y = torch.tensor(data["query_edge_y"], dtype=torch.long, device=dev).squeeze(-1)
            elif self.task_type == "Degree":
                query_x = torch.tensor(data["query_node_x"], dtype=torch.float32, device=dev)
                query_y = torch.tensor(data["query_node_y"], dtype=torch.long, device=dev).squeeze(-1)
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

        return (support_x, support_y, query_x, query_y)
