import os
import numpy as np
import torch
import logging
import gc

from torch.utils.data import Dataset
from typing import List, Tuple, Optional


logger = logging.getLogger("crane")


class SketchDataset(Dataset):
    def __init__(self, data: str | List, task_type: str = None, device: Optional[torch.device | str] = None, lazy: bool = False):
        super().__init__()
        self.name: str = ""
        self.task_type: str = task_type
        self.task_names: List[str] = []
        self.device = device
        self.lazy = lazy

        self._npz_paths: Optional[List[str]] = None
        self._cached_idx: Optional[int] = None
        self._cached_item: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None

        if isinstance(data, list):
            if lazy:
                logger.warning("lazy=True only takes effect when data is a directory path (str); "
                               "when passing a list, lazy mode is ignored and data is used as-is from memory.")
            self.data = data
            if self.device is None and len(data) > 0:
                self.device = data[0][0].device
            self.name = ""
            self.task_names = ["" for _ in range(len(self.data))]
        elif isinstance(data, str):
            root_path = data
            if lazy:
                self._init_lazy_from_dir(root_path)
                self.data = None
            else:
                self.data, self.name, self.task_names = self.load_npz_dataset(root_path, self.device)
        else:
            raise ValueError(f"Unknown data type: {type(data)}")

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
        else:
            return self.data[index]

    def __len__(self) -> int:
        if self.lazy:
            return len(self._npz_paths)
        return len(self.data)

    def release_cache(self):
        self._cached_idx = None
        self._cached_item = None
        gc.collect()

    def load_npz_dataset(self, root_path: str, device):
        name = os.path.basename(root_path)
        task_names, dir_list = self._scan_and_sort_task_dirs(root_path)

        dataset = []
        for folder in dir_list:
            folder_path = os.path.join(root_path, folder)
            npz_path = os.path.join(folder_path, "0.npz")
            if not os.path.isfile(npz_path):
                logger.warning(f"npz not found: {npz_path}")
                continue
            item = self._load_one_npz(npz_path, device_override=device)
            dataset.append(item)

        return dataset, name, task_names

    def _init_lazy_from_dir(self, root_path: str):
        self.name = os.path.basename(root_path)
        task_names, dir_list = self._scan_and_sort_task_dirs(root_path)
        npz_paths = []
        for folder in dir_list:
            npz_path = os.path.join(root_path, folder, "0.npz")
            if not os.path.isfile(npz_path):
                logger.warning(f"npz not found: {npz_path}")
                continue
            npz_paths.append(npz_path)
        self.task_names = task_names[:len(npz_paths)]
        self._npz_paths = npz_paths

    def _scan_and_sort_task_dirs(self, root_path: str):
        dir_list = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        try:
            dir_list = sorted(dir_list, key=lambda s: int(os.path.basename(s).split("_")[1]))
        except (IndexError, ValueError) as e:
            logger.warning(f"Task name sorting error: {dir_list}\nException: {e}")
            dir_list = sorted(dir_list)
        task_names = dir_list[:]
        return task_names, dir_list

    def _load_one_npz(self, npz_path: str, device_override=None):
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                support_x = torch.tensor(data["support_x"], dtype=torch.float32,
                                         device=(device_override if device_override is not None else self.device))
                support_y = torch.tensor(data["support_y"], dtype=torch.long,
                                         device=(device_override if device_override is not None else self.device))

                if self.task_type == "Basic":
                    query_x = torch.tensor(data["query_edge_x"], dtype=torch.float32,
                                           device=(device_override if device_override is not None else self.device))
                    query_y = torch.tensor(data["query_edge_y"], dtype=torch.long,
                                           device=(device_override if device_override is not None else self.device)).squeeze(-1)
                elif self.task_type == "Degree":
                    query_x = torch.tensor(data["query_node_x"], dtype=torch.float32,
                                           device=(device_override if device_override is not None else self.device))
                    query_y = torch.tensor(data["query_node_y"], dtype=torch.long,
                                           device=(device_override if device_override is not None else self.device)).squeeze(-1)
                else:
                    raise ValueError(f"Unknown task type: {self.task_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to load npz file: {npz_path}") from e

        return (support_x, support_y, query_x, query_y)


class MiniDataset(Dataset):
    # Each sample consists of support/query_x, support/query_y
    def __init__(self, x, y):
        super().__init__()
        self.data = [x, y]

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[0])
