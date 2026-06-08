"""Generate paper-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_LABELS: dict[str, str] = {
    "csp_svm": "CSP+SVM",
    "fbcsp_lda": "FBCSP+LDA",
    "riemann_mdm": "Riemann MDM",
    "riemann_ts_lr": "Riemann TS+LR",
    "eegnet": "EEGNet",
}

MODEL_ORDER: list[str] = [
    "csp_svm",
    "fbcsp_lda",
    "riemann_ts_lr",
    "riemann_mdm",
    "eegnet",
]

PANEL_TITLES: dict[str, str] = {
    "physionet": "PhysioNet MI",
    "bnci": "BNCI2014-001",
    "bnci2014_001": "BNCI2014-001",
}


def _dataset_key(value: str) -> str:
    v = str(value).lower()
    if "bnci" in v:
        return "bnci"
    if "physionet" in v:
        return "physionet"
    return v


def _panel_title(dataset_value: str, fallback: str = "") -> str:
    key = _dataset_key(dataset_value)
    return PANEL_TITLES.get(key, PANEL_TITLES.get(dataset_value, fallback or str(dataset_value)))


def _prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "balanced_accuracy_mean" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["model_key"] = out["model"].astype(str)
    out["model_label"] = out["model_key"].map(lambda m: MODEL_LABELS.get(m, m))
    out["use_ea"] = out["use_ea"].astype(bool)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    out["model_order"] = out["model_key"].map(lambda m: order.get(m, 99))
    out = out.sort_values(["model_order", "use_ea"])
    return out


def _plot_dataset_panel(ax, df: pd.DataFrame, title: str) -> None:
    prep = _prepare_summary(df)
    if prep.empty:
        ax.set_visible(False)
        return

    models = [m for m in MODEL_ORDER if m in set(prep["model_key"])]
    if not models:
        models = list(dict.fromkeys(prep["model_key"].tolist()))

    x = np.arange(len(models))
    width = 0.36
    no_ea_means, no_ea_stds, ea_means, ea_stds, labels = [], [], [], [], []

    for model in models:
        sub = prep[prep["model_key"] == model]
        no_row = sub[~sub["use_ea"]]
        ea_row = sub[sub["use_ea"]]
        labels.append(MODEL_LABELS.get(model, model))
        no_ea_means.append(float(no_row["balanced_accuracy_mean"].iloc[0]) if len(no_row) else np.nan)
        no_ea_stds.append(float(no_row["balanced_accuracy_std"].iloc[0]) if len(no_row) else 0.0)
        ea_means.append(float(ea_row["balanced_accuracy_mean"].iloc[0]) if len(ea_row) else np.nan)
        ea_stds.append(float(ea_row["balanced_accuracy_std"].iloc[0]) if len(ea_row) else 0.0)

    ax.bar(
        x - width / 2,
        no_ea_means,
        width,
        yerr=no_ea_stds,
        capsize=3,
        label="No EA",
        color="#9ebcda",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        ea_means,
        width,
        yerr=ea_stds,
        capsize=3,
        label="EA",
        color="#2c6b9e",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(title)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", alpha=0.25)


def plot_repeated_holdout_balanced_accuracy(
    physionet_summary: Path | None,
    bnci_summary: Path | None,
    out_dir: Path,
) -> Path | None:
    panels: list[tuple[pd.DataFrame, str]] = []
    if physionet_summary and physionet_summary.exists():
        phys_df = pd.read_csv(physionet_summary)
        if not phys_df.empty:
            ds = phys_df["dataset"].iloc[0] if "dataset" in phys_df.columns else "physionet"
            panels.append((phys_df, _panel_title(ds, "PhysioNet MI")))
    if bnci_summary and bnci_summary.exists():
        bnci_df = pd.read_csv(bnci_summary)
        if not bnci_df.empty:
            ds = bnci_df["dataset"].iloc[0] if "dataset" in bnci_df.columns else "bnci"
            panels.append((bnci_df, _panel_title(ds, "BNCI2014-001")))

    if not panels:
        return None

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    panel_letters = "ABCDEF"
    for ax, (df, title), letter in zip(axes, panels, panel_letters):
        _plot_dataset_panel(ax, df, f"({letter}) {title}")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Repeated subject-disjoint hold-out — balanced accuracy (mean ± std, 10 repetitions)", y=1.08, fontsize=12)
    fig.tight_layout()
    out = out_dir / "fig_repeated_holdout_accuracy"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def _ea_gain_from_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "balanced_accuracy_mean" not in df.columns:
        return pd.DataFrame()
    rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"].astype(str) == model]
        if sub.empty:
            continue
        no_row = sub[~sub["use_ea"].astype(bool)]
        ea_row = sub[sub["use_ea"].astype(bool)]
        if no_row.empty or ea_row.empty:
            continue
        rows.append({
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "delta_bal_acc": float(
                ea_row["balanced_accuracy_mean"].iloc[0] - no_row["balanced_accuracy_mean"].iloc[0]
            ),
        })
    return pd.DataFrame(rows)


def plot_ea_gain_balanced_accuracy(
    physionet_summary: Path | None,
    bnci_summary: Path | None,
    out_dir: Path,
    *,
    stem: str = "fig_ea_gain",
) -> Path | None:
    panels: list[tuple[pd.DataFrame, str]] = []
    if physionet_summary and physionet_summary.exists():
        gain = _ea_gain_from_summary(pd.read_csv(physionet_summary))
        if not gain.empty:
            panels.append((gain, "PhysioNet MI"))
    if bnci_summary and bnci_summary.exists():
        gain = _ea_gain_from_summary(pd.read_csv(bnci_summary))
        if not gain.empty:
            panels.append((gain, "BNCI2014-001"))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (gain, title), letter in zip(axes, panels, "AB"):
        colors = ["#2c6b9e" if v >= 0 else "#c44e52" for v in gain["delta_bal_acc"]]
        x = np.arange(len(gain))
        ax.bar(x, gain["delta_bal_acc"], color=colors, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_ylabel("Δ balanced accuracy (EA − no EA)")
        ax.set_title(f"({letter}) {title}")
        ax.set_xticks(x, gain["model_label"], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Euclidean Alignment gain by model", y=1.04, fontsize=12)
    fig.tight_layout()
    out = out_dir / stem
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def plot_ea_covariance_diagnostics(
    physionet_csv: Path | None,
    bnci_csv: Path | None,
    out_dir: Path,
    *,
    stem: str = "fig_ea_covariance_diagnostics",
) -> Path | None:
    panels: list[tuple[pd.Series, str]] = []
    for path, title in ((physionet_csv, "PhysioNet MI"), (bnci_csv, "BNCI2014-001")):
        if path and path.exists():
            row = pd.read_csv(path).iloc[0]
            panels.append((row, title))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    metrics = [
        ("within_before", "Within-subject\nbefore EA", "#d4a574"),
        ("within_after", "Within-subject\nafter EA", "#2c6b9e"),
        ("between_before", "Between-subject\nbefore EA", "#e8a87c"),
        ("between_after", "Between-subject\nafter EA", "#1a4f7a"),
    ]
    for ax, (row, title), letter in zip(axes, panels, "AB"):
        vals = [float(row[m]) for m, _, _ in metrics]
        labels = [lab for _, lab, _ in metrics]
        colors = [c for _, _, c in metrics]
        ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_title(f"({letter}) {title}")
        ax.set_ylabel("Mean covariance distance (Frobenius)")
        ax.tick_params(axis="x", rotation=15)
        w_red = row.get("within_reduction_pct", np.nan)
        b_red = row.get("between_reduction_pct", np.nan)
        ax.text(
            0.02, 0.98,
            f"Reduction: within {w_red:.1f}%, between {b_red:.1f}%",
            transform=ax.transAxes, va="top", fontsize=9,
        )
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("EA effect on trial covariance dispersion", y=1.02, fontsize=12)
    fig.tight_layout()
    out = out_dir / stem
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def plot_subject_level_variability(
    physionet_metrics: Path | None,
    bnci_metrics: Path | None,
    out_dir: Path,
    *,
    stem: str = "fig_subject_level_variability",
) -> Path | None:
    panels: list[tuple[pd.DataFrame, str]] = []
    for path, title in ((physionet_metrics, "PhysioNet MI"), (bnci_metrics, "BNCI2014-001")):
        if path and path.exists():
            df = pd.read_csv(path)
            if not df.empty and "accuracy" in df.columns:
                panels.append((df, title))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (df, title), letter in zip(axes, panels, "AB"):
        subj_acc = df.groupby("subject_id")["accuracy"].mean().to_numpy()
        bp = ax.boxplot(
            [subj_acc],
            widths=0.45,
            patch_artist=True,
            medianprops={"color": "#1a1a1a", "linewidth": 1.5},
        )
        bp["boxes"][0].set_facecolor("#9ebcda")
        bp["boxes"][0].set_alpha(0.85)
        ax.scatter(
            np.ones(len(subj_acc)) + np.random.default_rng(42).uniform(-0.08, 0.08, len(subj_acc)),
            subj_acc,
            alpha=0.35,
            s=12,
            color="#2c6b9e",
            edgecolors="none",
        )
        ax.set_xticks([1], ["Per-subject\nmean accuracy"])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"({letter}) {title}")
        ax.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylim(0.35, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.98, 0.02,
            f"n={len(subj_acc)} subjects\nμ={subj_acc.mean():.3f}, σ={subj_acc.std():.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        )

    fig.suptitle("Inter-subject accuracy variability (aggregated across model runs)", y=1.02, fontsize=12)
    fig.tight_layout()
    out = out_dir / stem
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def plot_lateralization_by_class(
    physionet_trials: Path | None,
    bnci_trials: Path | None,
    out_dir: Path,
    *,
    stem: str = "fig_lateralization_by_class",
) -> Path | None:
    panels: list[tuple[pd.DataFrame, str]] = []
    for path, title in ((physionet_trials, "PhysioNet MI"), (bnci_trials, "BNCI2014-001")):
        if path and path.exists():
            df = pd.read_csv(path)
            if not df.empty and "lateralization_index_mu" in df.columns:
                panels.append((df, title))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    class_colors = {"left_hand": "#2c6b9e", "right_hand": "#e8a87c"}
    for ax, (df, title), letter in zip(axes, panels, "AB"):
        for label, grp in df.groupby("label"):
            color = class_colors.get(str(label), "#888888")
            ax.hist(
                grp["lateralization_index_mu"].dropna(),
                bins=30,
                alpha=0.55,
                label=str(label).replace("_", " "),
                color=color,
                density=True,
            )
        ax.set_xlabel("Mu-band lateralization index")
        ax.set_ylabel("Density")
        ax.set_title(f"({letter}) {title}")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Mu-band power lateralization by imagined class (not baseline-corrected ERD/ERS)", y=1.03, fontsize=11)
    fig.tight_layout()
    out = out_dir / stem
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def plot_repeated_holdout_accuracy(summary_csv: Path, out_dir: Path) -> Path | None:
    """Backward-compatible wrapper: single-dataset plots are merged by the caller."""
    key = _dataset_key(summary_csv.parent.name)
    phys = summary_csv if key == "physionet" else None
    bnci = summary_csv if key == "bnci" else None
    return plot_repeated_holdout_balanced_accuracy(phys, bnci, out_dir)


def generate_all_paper_figures(results_root: Path, out_dir: Path) -> dict[str, Path | None]:
    """Generate the full manuscript figure set from publishable artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    phys_sum = results_root / "repeated_holdout" / "physionet" / "repeated_holdout_summary.csv"
    bnci_sum = results_root / "repeated_holdout" / "bnci" / "repeated_holdout_summary.csv"
    if not phys_sum.exists() or not bnci_sum.exists():
        found = {p.parent.name: p for p in results_root.glob("**/repeated_holdout_summary.csv")}
        phys_sum = found.get("physionet", phys_sum)
        bnci_sum = found.get("bnci", bnci_sum)

    paths = {
        "fig_repeated_holdout_accuracy": plot_repeated_holdout_balanced_accuracy(
            phys_sum if phys_sum.exists() else None,
            bnci_sum if bnci_sum.exists() else None,
            out_dir,
        ),
        "fig_ea_gain": plot_ea_gain_balanced_accuracy(
            phys_sum if phys_sum.exists() else None,
            bnci_sum if bnci_sum.exists() else None,
            out_dir,
        ),
        "fig_ea_covariance_diagnostics": plot_ea_covariance_diagnostics(
            results_root / "ea_diagnostics" / "physionet" / "ea_covariance_distances.csv",
            results_root / "ea_diagnostics" / "bnci" / "ea_covariance_distances.csv",
            out_dir,
        ),
        "fig_subject_level_variability": plot_subject_level_variability(
            results_root / "subject_level" / "physionet" / "subject_level_metrics.csv",
            results_root / "subject_level" / "bnci" / "subject_level_metrics.csv",
            out_dir,
        ),
        "fig_lateralization_by_class": plot_lateralization_by_class(
            results_root / "neurophysiology" / "physionet" / "erd_ers_trial_level.csv",
            results_root / "neurophysiology" / "bnci" / "erd_ers_trial_level.csv",
            out_dir,
        ),
    }
    write_figure_captions(out_dir)
    return paths


def write_figure_captions(out_dir: Path) -> None:
    captions = """# Figure captions

- **fig_repeated_holdout_accuracy**: (A) PhysioNet MI and (B) BNCI2014-001 — mean ± std balanced accuracy across 10 repeated subject-disjoint hold-out splits; grouped bars show no-EA vs EA per model.
- **fig_ea_gain**: Δ balanced accuracy (EA − no EA) by model; (A) PhysioNet, (B) BNCI2014-001.
- **fig_ea_covariance_diagnostics**: Trial covariance dispersion before and after EA; explains why covariance-based pipelines benefit most from alignment.
- **fig_subject_level_variability**: Distribution of per-subject mean accuracy (aggregated across model runs); inter-subject spread for discussion.
- **fig_lateralization_by_class**: Mu-band lateralization index distributions by imagined class (secondary physiological analysis; not baseline-corrected ERD/ERS).
- **fig_pipeline_workflow**: Subject-disjoint hold-out pipeline with shared preprocessing and EA ablation.
- **fig_model_ranking**: Model ranking under subject-independent evaluation.
- **fig_lateralization_vs_accuracy**: Correlation between mu-band lateralization and subject accuracy.
"""
    (out_dir / "figure_captions.md").write_text(captions, encoding="utf-8")
