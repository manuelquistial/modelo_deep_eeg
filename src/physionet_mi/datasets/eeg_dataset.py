"""PyTorch Dataset for trial-level EEG."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EEGDataset(Dataset):
    """X: (N, C, T), y: (N,) classes 0|1, groups: (N,) subject ids."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        trial_wise_normalize: bool = True,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.groups = np.asarray(groups)
        self.trial_wise_normalize = trial_wise_normalize

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.ascontiguousarray(self.X[idx], dtype=np.float32)).unsqueeze(0)
        if self.trial_wise_normalize:
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        s = torch.tensor(int(self.groups[idx]), dtype=torch.long)
        return x, y, s


def make_dataloader(
    dataset: EEGDataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
