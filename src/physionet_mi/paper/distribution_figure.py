"""Distribution plots for repeated hold-out balanced accuracy."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physionet_mi.paper.figure_common import (
    MODEL_LABELS,
    MODEL_ORDER,
    load_holdout_results,
    panel_title,
    save_figure,
)


def _plot_dataset_violin(ax, df: pd.DataFrame, title: str, letter: str) -> None:
    positions = []
    data = []
    colors = []
    tick_labels = []
    x = 0
    width = 0.35
    for model in MODEL_ORDER:
        for use_ea, offset, color in [(False, -width / 2, "#9ebcda"), (True, width / 2, "#2c6b9e")]:
            sub = df[(df["model"] == model) & (df["use_ea"] == use_ea)]
            if sub.empty:
                continue
            vals = sub["balanced_accuracy"].to_numpy()
            data.append(vals)
            positions.append(x + offset)
            colors.append(color)
            tick_labels.append("") if offset > 0 else tick_labels.append(MODEL_LABELS.get(model, model))
            # overlay points
            jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
            ax.scatter(
                np.full(len(vals), x + offset) + jitter,
                vals, s=18, color="#1a1a1a", alpha=0.55, zorder=3,
            )
        x += 1

    parts = ax.violinplot(data, positions=positions, widths=0.28, showmeans=False, showmedians=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor("white")
    for part in ("cbars", "cmins", "cmaxes"):
        if part in parts:
            parts[part].set_color("#444444")
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(f"({letter}) {title}")
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylim(0.4, 0.9)
    ax.grid(axis="y", alpha=0.25)


def generate_repeated_holdout_distributions(
    pub_root: Path,
    output_stem: Path,
) -> tuple[Path, Path] | None:
    panels = []
    for ds, letter in [("physionet", "A"), ("bnci", "B")]:
        df = load_holdout_results(pub_root, ds)
        if df is not None and not df.empty:
            title = panel_title(ds)
            panels.append((df, title, letter))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.5), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (df, title, letter) in zip(axes, panels):
        _plot_dataset_violin(ax, df, title, letter)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#9ebcda", lw=8, label="No EA"),
        Line2D([0], [0], color="#2c6b9e", lw=8, label="EA"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Repeated hold-out balanced accuracy distributions (10 repetitions)", y=1.06, fontsize=12)
    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def write_distribution_notes(path: Path) -> None:
    path.write_text(
        "# fig_repeated_holdout_distributions — notes\n\n"
        "- Source: `repeated_holdout_results.csv` (ok runs only).\n"
        "- Metric: balanced accuracy per repetition, model, and EA condition.\n"
        "- Violin + median line + jittered repetition points (n=10 each).\n"
        "- Complements bar-chart summary (`fig_repeated_holdout_accuracy`) by showing spread.\n",
        encoding="utf-8",
    )
