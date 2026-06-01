"""EEGMeModel: local CNN + Transformer (ported from funtions.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physionet_mi.config import ExperimentConfig


class LocalFeatureLearner(nn.Module):
    def __init__(self, in_channels: int = 1, eeg_channels: int = 64, F1: int = 7, dropout: float = 0.25):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels, F1, kernel_size=(1, 10), padding=(0, 5)),
            nn.BatchNorm2d(F1),
            nn.ELU(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1, kernel_size=(eeg_channels, 1)),
            nn.BatchNorm2d(F1),
            nn.ELU(),
        )
        self.pool = nn.AvgPool2d(kernel_size=(1, 10))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class EEGMeModel(nn.Module):
    def __init__(
        self,
        C: int = 64,
        T: int = 480,
        f: int = 7,
        e: int = 32,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        if T % 10 != 0:
            raise ValueError(f"n_times must be divisible by 10 (pooling), got T={T}")

        self.C = C
        self.T = T
        self.local = LocalFeatureLearner(in_channels=1, eeg_channels=C, F1=f, dropout=dropout)
        self.proj = nn.Conv2d(f, e, kernel_size=(1, 1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=e,
            nhead=4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.embedding_dim = e
        self.fc = nn.Linear(e, num_classes)

    def forward(self, x: torch.Tensor):
        local_features = self.local(x)
        x = self.proj(local_features)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        global_features = self.transformer(x)
        x = global_features
        x = torch.mean(x, dim=1)
        embeddings = F.normalize(x, p=2, dim=1)
        logits = self.fc(embeddings)
        return logits, embeddings, local_features, global_features


def build_model(cfg: ExperimentConfig) -> EEGMeModel:
    return EEGMeModel(
        C=cfg.model.n_channels,
        T=cfg.model.n_times,
        f=cfg.model.f1,
        e=cfg.model.embed_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
