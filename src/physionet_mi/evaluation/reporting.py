"""Save metrics, plots, and run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from physionet_mi.constants import CLASS_TO_NAME
from physionet_mi.evaluation.metrics import (
    classification_report_fixed,
    compute_metrics,
    confusion_matrix_fixed,
)

# Keys with large/non-JSON payloads kept for CSV/plots only.
_METRICS_JSON_EXCLUDE = frozenset({"groups"})


def _json_sanitize(value: Any) -> Any:
    """Convert numpy/scalar values to JSON-serializable Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _prepare_metrics_for_json(metrics: dict[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in metrics.items() if k not in _METRICS_JSON_EXCLUDE}
    return _json_sanitize(payload)


def save_run_artifacts(
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg_dict: dict,
    history: list[dict] | None = None,
    extra: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred)
    if extra:
        metrics.update(extra)

    (out_dir / "metrics.json").write_text(
        json.dumps(_prepare_metrics_for_json(metrics), indent=2),
        encoding="utf-8",
    )
    (out_dir / "classification_report.txt").write_text(
        classification_report_fixed(y_true, y_pred), encoding="utf-8"
    )
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8"
    )

    pred_dict: dict = {"y_true": y_true, "y_pred": y_pred}
    if extra and "groups" in extra:
        pred_dict["subject_id"] = extra["groups"]
    pd.DataFrame(pred_dict).to_csv(out_dir / "predictions.csv", index=False)

    if history:
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    cm = confusion_matrix_fixed(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = [CLASS_TO_NAME[i] for i in sorted(CLASS_TO_NAME)]
    ax.set_xticks(range(2), labels, rotation=30, ha="right")
    ax.set_yticks(range(2), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    return metrics
