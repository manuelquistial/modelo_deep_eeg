"""Subject-level metrics from prediction files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from physionet_mi.constants import CLASS_ORDER


def _subject_metrics(y_true, y_pred) -> dict:
    acc = float((y_true == y_pred).mean()) if len(y_true) else np.nan
    bal = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else np.nan
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    left_mask = y_true == CLASS_ORDER[0]
    right_mask = y_true == CLASS_ORDER[1]
    left_recall = float((y_pred[left_mask] == y_true[left_mask]).mean()) if left_mask.any() else np.nan
    right_recall = float((y_pred[right_mask] == y_true[right_mask]).mean()) if right_mask.any() else np.nan
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_f1": f1,
        "left_recall": left_recall,
        "right_recall": right_recall,
        "false_left_as_right": int(np.sum((y_true == CLASS_ORDER[0]) & (y_pred == CLASS_ORDER[1]))),
        "false_right_as_left": int(np.sum((y_true == CLASS_ORDER[1]) & (y_pred == CLASS_ORDER[0]))),
    }


def collect_subject_level_metrics(predictions_root: Path) -> pd.DataFrame:
    rows = []
    for pred_path in predictions_root.rglob("predictions.csv"):
        df = pd.read_csv(pred_path)
        if "subject_id" not in df.columns:
            continue
        run_dir = pred_path.parent
        parts = run_dir.name.split("_")
        model = parts[0] if parts else "unknown"
        use_ea = "ea" in run_dir.name and "no_ea" not in run_dir.name
        seed = None
        for p in parts:
            if p.startswith("seed"):
                seed = int(p.replace("seed", ""))
        for sid, grp in df.groupby("subject_id"):
            m = _subject_metrics(grp["y_true"].to_numpy(), grp["y_pred"].to_numpy())
            rows.append({
                "run_dir": str(run_dir),
                "model": model,
                "use_ea": use_ea,
                "seed": seed,
                "subject_id": int(sid),
                "n_trials": len(grp),
                **m,
            })
    return pd.DataFrame(rows)


def plot_subject_level_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = df.groupby(["model", "use_ea"])["accuracy"].apply(list)
    labels, data = zip(*[(f"{m}\nEA={ea}", v) for (m, ea), v in groups.items()]) if len(groups) else ([], [])
    if data:
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("Subject accuracy")
        ax.set_title("Subject-level accuracy distribution")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def write_subject_level_summary(df: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Subject-level analysis summary\n"]
    if df.empty:
        lines.append("Insufficient evidence: no prediction files with subject_id.\n")
    else:
        hardest = df.groupby("subject_id")["accuracy"].mean().sort_values().head(10)
        lines.append("## Hardest subjects (mean accuracy)\n")
        for sid, acc in hardest.items():
            lines.append(f"- Subject {sid}: {acc:.3f}\n")
        lines.append(f"\nTotal subject-run rows: {len(df)}\n")
    out_path.write_text("".join(lines), encoding="utf-8")
