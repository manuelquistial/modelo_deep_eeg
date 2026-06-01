"""Classification metrics (aligned with workshop)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from physionet_mi.constants import CLASS_ORDER, CLASS_TO_NAME


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def confusion_matrix_fixed(y_true: np.ndarray, y_pred: np.ndarray, normalize: str | None = None):
    return confusion_matrix(y_true, y_pred, labels=CLASS_ORDER, normalize=normalize)


def classification_report_fixed(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    target_names = [CLASS_TO_NAME[c] for c in CLASS_ORDER]
    return classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        target_names=target_names,
        digits=3,
        zero_division=0,
    )


@torch.no_grad()
def predict_loader(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels, all_groups = [], [], []
    for x, y, s in loader:
        x = x.to(device)
        logits, *_ = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
        all_groups.append(s.numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_groups),
    )
