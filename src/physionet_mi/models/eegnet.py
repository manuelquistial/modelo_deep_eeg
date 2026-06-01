"""EEGNet — Lawhern et al. 2018 (ported from modelo_bilstm).

Input: (batch, 1, C, T) — same tensor layout as EEGMeModel / EEGDataset.
Output: (logits, embeddings) for trainer compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physionet_mi.config import ExperimentConfig


class EEGNet(nn.Module):
    """EEGNet for trial-level (C, T) inputs wrapped as (1, C, T)."""

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 480,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.F1, self.D, self.F2 = F1, D, F2
        pad_t = kernel_length // 2
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, pad_t), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.sep_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            flat = self._forward_features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.fc = nn.Linear(flat, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sep_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor):
        """Return (logits, embeddings) like EEGMeModel."""
        feat = self._forward_features(x)
        embeddings = F.normalize(feat, p=2, dim=1)
        logits = self.fc(feat)
        return logits, embeddings


def build_eegnet(cfg: ExperimentConfig) -> EEGNet:
    m = cfg.model
    return EEGNet(
        n_channels=m.n_channels,
        n_times=m.n_times,
        F1=m.eegnet_F1,
        D=m.eegnet_D,
        F2=m.eegnet_F2,
        kernel_length=m.eegnet_kernel_length,
        dropout=m.eegnet_dropout,
        num_classes=m.num_classes,
    )
