"""Publication-quality pipeline architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from physionet_mi.paper.figure_common import save_figure


def _box(ax, xy, w, h, text, fc="#f7f9fc", ec="#2c3e50", fontsize=8):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def generate_pipeline_architecture_figure(output_stem: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Row 1: datasets
    _box(ax, (0.3, 5.5), 2.2, 0.9, "PhysioNet MI\n108 subjects, 64 ch", fc="#e8f4fc")
    _box(ax, (2.8, 5.5), 2.2, 0.9, "BNCI2014-001\n9 subjects, 22 ch", fc="#e8f4fc")

    # Preprocessing
    _box(ax, (5.5, 5.2), 3.6, 1.5,
         "Preprocessing\n• outlier rejection\n• high-pass 4 Hz\n• time harmonization\n• binary L/R MI",
         fc="#fff8e6")

    # EA branch
    _box(ax, (9.5, 5.5), 2.0, 0.9, "Euclidean Alignment\n(optional ablation)", fc="#fdebd0", ec="#c0392b")
    _box(ax, (9.5, 4.3), 2.0, 0.9, "No EA\n(baseline)", fc="#f5f5f5")

    # Models
    models_y = 2.8
    model_w, model_h = 1.55, 0.75
    models = [
        ("CSP+SVM", "#d6eaf8"),
        ("FBCSP+LDA", "#d6eaf8"),
        ("Riemann MDM", "#e8daef"),
        ("Riemann TS+LR", "#e8daef"),
        ("EEGNet", "#d5f5e3"),
    ]
    x0 = 0.4
    for i, (name, color) in enumerate(models):
        _box(ax, (x0 + i * 1.7, models_y), model_w, model_h, name, fc=color, fontsize=7.5)

    # Evaluation
    _box(ax, (9.0, 2.5), 4.5, 1.2,
         "Repeated subject-disjoint hold-out\n"
         "master_seed=42, n_repeats=10\n"
         "strict DEV/TEST subject separation\n"
         "VAL inside DEV (deep models)\n"
         "metrics: balanced acc., macro-F1, κ",
         fc="#eafaf1", ec="#1e8449")

    # Stats + analyses
    _box(ax, (0.4, 0.5), 4.0, 1.5,
         "Statistical analysis\n• Wilcoxon (EA vs no-EA)\n• Friedman + Holm post-hoc\n• bootstrap 95% CI",
         fc="#f4ecf7")
    _box(ax, (4.7, 0.5), 4.0, 1.5,
         "Complementary analyses\n• EA covariance diagnostics\n• subject-level variability\n• mu-band lateralization",
         fc="#f4ecf7")
    _box(ax, (9.0, 0.5), 4.5, 1.5,
         "Outputs\nartifacts/runs/publishable/\nartifacts/paper/tables & figures",
         fc="#f8f9f9")

    arrows = [
        ((1.4, 5.5), (5.5, 5.9)),
        ((3.9, 5.5), (5.5, 5.9)),
        ((9.1, 5.2), (9.5, 5.9)),
        ((9.1, 5.2), (9.5, 4.7)),
        ((7.3, 5.2), (1.2, 3.55)),
        ((10.5, 4.3), (10.5, 3.7)),
        ((6.0, 2.8), (9.0, 3.1)),
        ((11.2, 2.5), (2.4, 2.0)),
        ((11.2, 2.5), (6.7, 2.0)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.0, color="#566573",
            connectionstyle="arc3,rad=0.1",
        ))

    ax.text(7.0, 6.6, "EEG Motor Imagery Benchmark Pipeline", ha="center", fontsize=14, fontweight="bold")
    ax.text(7.0, 6.25, "Subject-disjoint evaluation — no trial leakage across train/test subjects",
            ha="center", fontsize=9, color="#566573")

    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def write_pipeline_notes(path: Path) -> None:
    path.write_text(
        "# fig_pipeline_architecture — methodological notes\n\n"
        "- Diagram summarizes the publishable benchmark workflow; not a data-flow UML diagram.\n"
        "- **Subject-disjoint hold-out:** train/val/test subjects are disjoint; repeated with 10 seeds from master_seed=42.\n"
        "- **EA ablation:** same splits; preprocessing differs only by Euclidean Alignment on/off.\n"
        "- **Deep models:** validation split carved from DEV subjects only.\n"
        "- **Excluded:** leave-one-subject-out (not part of this benchmark).\n",
        encoding="utf-8",
    )
