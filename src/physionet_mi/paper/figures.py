"""Generate paper-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_repeated_holdout_accuracy(summary_csv: Path, out_dir: Path) -> Path | None:
    if not summary_csv.exists():
        return None
    df = pd.read_csv(summary_csv)
    if df.empty or "accuracy_mean" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r['model']}\nEA={r['use_ea']}" for _, r in df.iterrows()]
    means = df["accuracy_mean"].to_numpy()
    stds = df.get("accuracy_std", pd.Series([0] * len(df))).to_numpy()
    ax.bar(range(len(df)), means, yerr=stds, capsize=4)
    ax.set_xticks(range(len(df)), labels, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Repeated hold-out accuracy (mean ± std)")
    fig.tight_layout()
    out = out_dir / "fig_repeated_holdout_accuracy"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out.with_suffix(".png")


def write_figure_captions(out_dir: Path) -> None:
    captions = """# Figure captions

- **fig_pipeline_workflow**: Subject-disjoint hold-out pipeline with shared preprocessing and EA ablation.
- **fig_repeated_holdout_accuracy**: Mean ± std accuracy across repeated subject-disjoint splits.
- **fig_ea_gain**: Balanced accuracy gain from Euclidean Alignment by model and dataset.
- **fig_model_ranking**: Model ranking under subject-independent evaluation.
- **fig_subject_level_heatmap**: Per-subject accuracy across models.
- **fig_lateralization_vs_accuracy**: Correlation between mu-band lateralization and subject accuracy.
"""
    (out_dir / "figure_captions.md").write_text(captions, encoding="utf-8")
