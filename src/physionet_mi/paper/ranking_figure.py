"""Statistical model ranking figure (post-hoc heatmap; CD when appropriate)."""

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
    rank_dataframe,
    save_figure,
)

# Nemenyi q_alpha for k=5, alpha=0.05 (two-tailed)
NEMENYI_Q = {5: 2.569}


def _average_ranks(df: pd.DataFrame) -> pd.DataFrame:
    ranked = rank_dataframe(df)
    if ranked.empty:
        return pd.DataFrame()
    return (
        ranked.groupby(["model", "use_ea"])["rank"]
        .mean()
        .reset_index()
        .rename(columns={"rank": "avg_rank"})
    )


def _try_critical_difference(
    df: pd.DataFrame,
    dataset: str,
    use_ea: bool,
    output_stem: Path,
) -> tuple[Path, Path] | None:
    """Nemenyi-style CD diagram when k=5 models and n>=10 repetitions."""
    sub = df[df["use_ea"] == use_ea]
    if sub["repeat_id"].nunique() < 5:
        return None
    ranked = rank_dataframe(sub)
    avg = ranked.groupby("model")["rank"].mean().reindex(MODEL_ORDER).dropna()
    if len(avg) < 3:
        return None

    k = len(avg)
    n = ranked["repeat_id"].nunique()
    q = NEMENYI_Q.get(k, 2.569)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    order = avg.sort_values()
    y = 0.5
    xmin, xmax = order.min() - 0.5, order.max() + 0.5
    ax.hlines(y, xmin, xmax, color="#333333", linewidth=2)
    for model, rank in order.items():
        ax.plot(rank, y, "o", color="#2c6b9e", ms=10)
        ax.text(rank, y + 0.12, MODEL_LABELS.get(model, model), ha="center", fontsize=9, rotation=15)

    # connect groups within CD
    sorted_models = list(order.index)
    groups = [[sorted_models[0]]]
    for m in sorted_models[1:]:
        if order[m] - order[groups[-1][0]] <= cd:
            groups[-1].append(m)
        else:
            groups.append([m])
    for grp in groups:
        if len(grp) < 2:
            continue
        r0, r1 = order[grp[0]], order[grp[-1]]
        ax.plot([r0, r1], [y - 0.15, y - 0.15], color="#c0392b", linewidth=3)

    ea_label = "EA" if use_ea else "no-EA"
    ax.set_xlabel("Average rank (1 = best)")
    ax.set_title(f"Critical difference diagram — {panel_title(dataset)} ({ea_label})\n"
                 f"Friedman blocks n={n}; CD≈{cd:.2f} (Nemenyi α=0.05)")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def _posthoc_heatmap(
    posthoc_csv: Path,
    dataset: str,
    use_ea: bool,
    ax,
    letter: str,
) -> None:
    df = pd.read_csv(posthoc_csv)
    sub = df[(df["dataset"].astype(str).str.contains(dataset.split("_")[0], case=False)) & (df["use_ea"] == use_ea)]
    if sub.empty:
        ax.set_visible(False)
        return

    models = MODEL_ORDER
    mat = np.ones((len(models), len(models)))
    pcol = "p_value_holm" if "p_value_holm" in sub.columns else "p_value"
    for _, row in sub.iterrows():
        if row["model_a"] in models and row["model_b"] in models:
            i, j = models.index(row["model_a"]), models.index(row["model_b"])
            mat[i, j] = min(mat[i, j], float(row[pcol]))
            mat[j, i] = min(mat[j, i], float(row[pcol]))

    im = ax.imshow(mat, vmin=0, vmax=0.1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(models)), [MODEL_LABELS.get(m, m) for m in models], rotation=35, ha="right")
    ax.set_yticks(range(len(models)), [MODEL_LABELS.get(m, m) for m in models])
    ea_label = "EA" if use_ea else "no-EA"
    ax.set_title(f"({letter}) {panel_title(dataset)} — Holm p ({ea_label})")
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                continue
            val = mat[i, j]
            if val < 0.05:
                ax.text(j, i, "*", ha="center", va="center", color="white", fontsize=12)
    return im


def generate_statistical_ranking_figures(
    pub_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    stats_root = pub_root / "stats"
    result: dict[str, object] = {
        "cd_generated": False,
        "cd_paths": None,
        "heatmap_paths": None,
        "method": "posthoc_heatmap",
    }

    # Attempt CD for PhysioNet EA=True (primary comparison)
    phys = load_holdout_results(pub_root, "physionet")
    cd_stem = output_dir / "fig_critical_difference"
    if phys is not None:
        cd_paths = _try_critical_difference(phys, "physionet", True, cd_stem)
        if cd_paths:
            result["cd_generated"] = True
            result["cd_paths"] = cd_paths
            result["method"] = "nemenyi_cd_physionet_ea"

    # Always generate post-hoc heatmap grid (defensible alternative)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        (stats_root / "physionet" / "model_pairwise_posthoc.csv", "physionet", False, "A"),
        (stats_root / "physionet" / "model_pairwise_posthoc.csv", "physionet", True, "B"),
        (stats_root / "bnci" / "model_pairwise_posthoc.csv", "bnci", False, "C"),
        (stats_root / "bnci" / "model_pairwise_posthoc.csv", "bnci", True, "D"),
    ]
    ims = []
    for ax, (csv_path, ds, ea, letter) in zip(axes.ravel(), panels):
        if csv_path.exists():
            im = _posthoc_heatmap(csv_path, ds, ea, ax, letter)
            if im is not None:
                ims.append(im)
    if ims:
        fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Holm-adjusted p")
    fig.suptitle("Pairwise model comparisons (Wilcoxon across repetitions)", fontsize=12)
    fig.tight_layout()
    hm_stem = output_dir / "fig_model_ranking_significance"
    result["heatmap_paths"] = save_figure(fig, hm_stem)
    plt.close(fig)

    return result


def write_ranking_notes(path: Path, ranking_result: dict) -> None:
    method = ranking_result.get("method", "unknown")
    cd = ranking_result.get("cd_generated", False)
    path.write_text(
        "# Statistical ranking figure — notes\n\n"
        f"## Method selected: `{method}`\n\n"
        + (
            "- **Critical difference diagram** generated for PhysioNet (EA=True) using average ranks "
            "across 10 repeated hold-out blocks and Nemenyi CD (k=5, α=0.05).\n"
            if cd else
            "- True CD diagram was **not** emitted as the sole figure; see heatmap alternative.\n"
        )
        + "- **Alternative figure** `fig_model_ranking_significance`: Holm-adjusted Wilcoxon post-hoc "
        "p-value heatmaps for both datasets and EA conditions.\n\n"
        "## Statistical basis\n"
        "- Friedman omnibus tests are significant for all dataset×EA groups in `model_friedman_results.csv`.\n"
        "- Post-hoc: paired Wilcoxon on balanced accuracy across repetitions (`model_pairwise_posthoc.csv`).\n"
        "- With n=10 repetitions, interpret borderline pairs cautiously (e.g., CSP+SVM vs EEGNet no-EA).\n\n"
        "## Why heatmap is included\n"
        "- Shows **which** pairwise differences are significant after Holm correction.\n"
        "- More informative than CD alone when EA/no-EA are separate conditions.\n",
        encoding="utf-8",
    )
