"""Reusable training loop with early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from physionet_mi.config import ExperimentConfig
from physionet_mi.utils.device import get_device


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path
    history: list[dict]


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: ExperimentConfig,
    out_dir: Path,
) -> TrainResult:
    device = get_device(cfg.train.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.scheduler_factor,
        patience=cfg.train.scheduler_patience,
    )
    criterion = nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoints" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(cfg.train.max_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            logits, *_ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        val_loss = train_loss
        if val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for x, y, _ in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits, *_ = model(x)
                    val_loss += criterion(logits, y).item()
                    n_val += 1
            val_loss /= max(n_val, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= cfg.train.early_stopping_patience:
            break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        checkpoint_path=ckpt_path,
        history=history,
    )
