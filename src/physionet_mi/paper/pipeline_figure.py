"""Publication-quality pipeline architecture diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from physionet_mi.paper.figure_common import save_figure

_STAGE_STYLE = dict(fontsize=7, color="#5d6d7e", fontweight="bold", ha="center", va="bottom")
_BOX_EC = "#2c3e50"
_ARROW_KW = dict(arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#566573")


def _box(
    ax,
    xy,
    w,
    h,
    text,
    *,
    fc="#f7f9fc",
    ec=_BOX_EC,
    fontsize=7.5,
    lw=1.1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.25,
    )
    return patch


def _stage(ax, x, w, y, label: str) -> None:
    ax.text(x + w / 2, y, label, **_STAGE_STYLE)


def _arrow_h(ax, x1: float, x2: float, y: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            connectionstyle="arc3,rad=0",
            **_ARROW_KW,
        )
    )


def _arrow_v(ax, x: float, y1: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x, y1),
            (x, y2),
            connectionstyle="arc3,rad=0",
            **_ARROW_KW,
        )
    )


def _arrow_path(ax, points: list[tuple[float, float]]) -> None:
    for start, end in zip(points[:-1], points[1:]):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle="arc3,rad=0",
                **_ARROW_KW,
            )
        )


def generate_pipeline_architecture_figure(output_stem: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    stage_y = 4.05
    mid_y = 2.35

    # --- Stage 1: datasets ---
    ds_x, ds_w = 0.25, 1.55
    _stage(ax, ds_x, ds_w, stage_y, "Data")
    phys_h = _box(
        ax,
        (ds_x, mid_y + 0.55),
        ds_w,
        0.72,
        "PhysioNet MI\n108 subj., 64 ch",
        fc="#e8f4fc",
    )
    bnci_h = _box(
        ax,
        (ds_x, mid_y - 0.55),
        ds_w,
        0.72,
        "BNCI2014-001\n9 subj., 22 ch",
        fc="#e8f4fc",
    )

    # --- Stage 2: preprocessing ---
    pre_x, pre_w = 2.2, 1.55
    _stage(ax, pre_x, pre_w, stage_y, "Preproc.")
    pre = _box(
        ax,
        (pre_x, mid_y - 0.55),
        pre_w,
        1.45,
        "Preprocessing\n• outlier rejection\n• 4 Hz high-pass\n• time harmonization\n• binary L/R MI",
        fc="#fff8e6",
    )

    # --- Stage 3: subject-disjoint splits ---
    split_x, split_w = 4.15, 1.75
    _stage(ax, split_x, split_w, stage_y, "Splits")
    split = _box(
        ax,
        (split_x, mid_y - 0.62),
        split_w,
        1.58,
        "Subject-disjoint\npartitions\nDEV: train / val\nTEST: held-out\n(no overlap)",
        fc="#ebf5fb",
        ec="#2874a6",
    )

    # --- Stage 4: EA ablation branch (parallel, side-by-side) ---
    no_ea_x, ea_x, ab_w = 6.5, 8.05, 1.15
    ab_group_w = ea_x + ab_w - no_ea_x
    no_ea_y = mid_y + 0.18 - 0.31
    ea_y = mid_y - 0.18 - 0.31
    _stage(ax, no_ea_x - 0.1, ab_group_w + 0.2, stage_y, "EA branch")
    ablation = Rectangle(
        (no_ea_x - 0.1, ea_y - 0.08),
        ab_group_w + 0.2,
        (no_ea_y + 0.62) - (ea_y - 0.08) + 0.16,
        linewidth=1.0,
        edgecolor="#c0392b",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(ablation)
    ax.text(no_ea_x + ab_group_w / 2, no_ea_y + 0.78, "Ablation", ha="center", fontsize=6.5, color="#c0392b")
    _box(ax, (no_ea_x, no_ea_y), ab_w, 0.62, "No EA", fc="#f5f5f5", ec="#7f8c8d")
    _box(ax, (ea_x, ea_y), ab_w, 0.62, "EA", fc="#fdebd0", ec="#c0392b")

    # --- Stage 5: decoders ---
    mod_x, mod_w = 10.0, 1.65
    _stage(ax, mod_x, mod_w, stage_y, "Models")
    models = _box(
        ax,
        (mod_x, mid_y - 0.72),
        mod_w,
        1.78,
        "Decoders\nCSP+SVM · FBCSP+LDA\nRiemann MDM · TS+LR\nEEGNet",
        fc="#eaf2f8",
    )

    # --- Stage 6: evaluation ---
    ev_x, ev_w = 12.1, 1.85
    _stage(ax, ev_x, ev_w, stage_y, "Evaluation")
    eval_box = _box(
        ax,
        (ev_x, mid_y - 0.62),
        ev_w,
        1.58,
        "Repeated hold-out\n10 repetitions\n(master seed 42)\nBal. acc., macro-F1, κ",
        fc="#eafaf1",
        ec="#1e8449",
    )

    # --- Stage 7: statistical analysis ---
    st_x, st_w = 14.4, 1.85
    _stage(ax, st_x, st_w, stage_y, "Statistics")
    stats = _box(
        ax,
        (st_x, mid_y - 0.62),
        st_w,
        1.58,
        "Statistical analysis\n• Wilcoxon (EA vs no-EA)\n• Friedman + Holm\n• bootstrap 95% CI",
        fc="#f4ecf7",
        ec="#7d3c98",
    )

    # --- Auxiliary analyses (off main path) ---
    aux = _box(
        ax,
        (12.1, 0.35),
        4.15,
        0.72,
        "Auxiliary analyses: EA covariance diagnostics, subject variability, mu-band lateralization",
        fc="#fbfcfc",
        ec="#aeb6bf",
        fontsize=6.5,
    )

    # --- Arrows (orthogonal only; stay below title band) ---
    ds_right = ds_x + ds_w
    pre_right = pre_x + pre_w
    split_right = split_x + split_w
    no_ea_right = no_ea_x + ab_w
    ea_right = ea_x + ab_w
    mod_right = mod_x + mod_w
    ev_right = ev_x + ev_w
    trunk_y = mid_y
    no_ea_lane = mid_y + 0.18
    ea_lane = mid_y - 0.18

    _arrow_h(ax, ds_right, pre_x, mid_y + 0.91)
    _arrow_h(ax, ds_right, pre_x, mid_y - 0.19)

    _arrow_h(ax, pre_right, split_x, trunk_y)

    fork_x = split_right + 0.25
    merge_x = ea_right + 0.5
    _arrow_path(ax, [(split_right, trunk_y), (fork_x, trunk_y)])
    _arrow_path(ax, [(fork_x, trunk_y), (fork_x, no_ea_lane), (no_ea_x, no_ea_lane)])
    _arrow_path(ax, [(fork_x, trunk_y), (fork_x, ea_lane), (ea_x, ea_lane)])
    _arrow_path(
        ax,
        [
            (no_ea_right, no_ea_lane),
            (merge_x, no_ea_lane),
            (merge_x, trunk_y),
            (mod_x, trunk_y),
        ],
    )
    _arrow_path(ax, [(ea_right, ea_lane), (merge_x, ea_lane), (merge_x, trunk_y)])

    _arrow_h(ax, mod_right, ev_x, trunk_y)
    _arrow_h(ax, ev_right, st_x, trunk_y)

    # Dashed link from evaluation to auxiliary (no crossing main flow)
    _arrow_path(
        ax,
        [
            (ev_x + ev_w / 2, mid_y - 0.62),
            (ev_x + ev_w / 2, 1.07),
        ],
    )
    ax.plot(
        [ev_x + ev_w / 2, st_x + st_w / 2],
        [1.07, 1.07],
        linestyle="--",
        color="#aeb6bf",
        linewidth=0.9,
    )

    ax.text(
        8.6,
        4.55,
        "EEG Motor Imagery Benchmark Pipeline",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        8.6,
        4.22,
        "Strict subject-disjoint DEV/TEST evaluation — no trial leakage across subjects",
        ha="center",
        fontsize=8,
        color="#566573",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def write_pipeline_notes(path: Path) -> None:
    path.write_text(
        "# fig_pipeline_architecture — methodological notes\n\n"
        "- Left-to-right workflow diagram for the publishable benchmark.\n"
        "- **Subject-disjoint hold-out:** DEV and TEST subjects never overlap; "
        "deep models use validation subjects carved from DEV only.\n"
        "- **EA ablation:** parallel No-EA and EA branches share identical subject splits.\n"
        "- **Decoders:** CSP+SVM, FBCSP+LDA, Riemann MDM, Riemann TS+LR, and EEGNet.\n"
        "- **Evaluation:** ten repeated hold-outs (master seed 42); balanced accuracy, macro-F1, κ.\n"
        "- **Statistics:** Wilcoxon (EA vs no-EA), Friedman with Holm post-hoc, bootstrap 95% CI.\n"
        "- **Excluded:** leave-one-subject-out (not part of this benchmark).\n",
        encoding="utf-8",
    )
