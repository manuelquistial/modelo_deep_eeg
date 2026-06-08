"""Audit existing vs missing manuscript figures."""

from __future__ import annotations

from pathlib import Path


def audit_figures(figures_dir: Path, pub_root: Path) -> str:
    existing = {p.stem for p in figures_dir.glob("*.png")} if figures_dir.exists() else set()

    rows = [
        ("fig_repeated_holdout_accuracy", "existing", "reusable", "Primary bar chart; regenerate with panel labels"),
        ("fig_ea_gain", "partial", "reusable", "In manuscript_bundle; copy to paper/figures"),
        ("fig_ea_covariance_diagnostics", "partial", "reusable", "In manuscript_bundle; copy to paper/figures"),
        ("fig_subject_level_variability", "partial", "reusable", "In manuscript_bundle; copy to paper/figures"),
        ("fig_lateralization_by_class", "partial", "reusable", "In manuscript_bundle; copy to paper/figures"),
        ("fig_pipeline_architecture", "missing", "create", "Master workflow diagram"),
        ("fig_topoplots_motor_imagery", "missing", "create", "MNE mu-band 2x2 topomaps"),
        ("fig_topoplots_before_after_ea", "missing", "create", "PhysioNet EA spatial comparison"),
        ("fig_repeated_holdout_distributions", "missing", "create", "Violin/box per-repetition distributions"),
        ("fig_critical_difference", "missing", "create", "Nemenyi CD if supported"),
        ("fig_model_ranking_significance", "missing", "create", "Post-hoc heatmap alternative"),
        ("lateralization_distribution_by_class (per-dataset)", "existing", "legacy", "Under neurophysiology/; superseded by fig_lateralization_by_class"),
    ]

    lines = [
        "# Figure gap analysis\n\n",
        f"Audited: `{figures_dir}` and `{pub_root}`\n\n",
        "## Summary\n\n",
        f"- PNG figures currently in `artifacts/paper/figures/`: **{len(existing)}**\n",
        "- **Reusable:** repeated hold-out bar, EA gain, covariance, subject-level, lateralization (from bundle)\n",
        "- **Missing (this task):** pipeline architecture, topomaps, repetition distributions, statistical ranking\n\n",
        "| Figure | Status | Action | Notes |\n",
        "| --- | --- | --- | --- |\n",
    ]
    for fig, status, action, notes in rows:
        in_dir = "yes" if fig.split()[0] in existing else "no"
        lines.append(f"| {fig} | {status} | {action} | {notes} (in paper/figures: {in_dir}) |\n")

    lines.append("\n## Data sources verified\n\n")
    checks = [
        pub_root / "repeated_holdout/physionet/repeated_holdout_results.csv",
        pub_root / "repeated_holdout/bnci/repeated_holdout_results.csv",
        pub_root / "stats/physionet/model_pairwise_posthoc.csv",
        pub_root / "neurophysiology/physionet/erd_ers_trial_level.csv",
        pub_root / "ea_diagnostics/physionet/ea_covariance_distances.csv",
        Path("artifacts/cache/physionet_lr_no_ea_108sub_all/meta.json"),
    ]
    for p in checks:
        rel = p.as_posix()
        ok = "Available" if p.exists() else "**Missing**"
        lines.append(f"- `{rel}`: {ok}\n")

    lines.append("\n## Topoplot limitations\n\n")
    lines.append(
        "- PhysioNet: full 64-channel cache available for rigorous mu-band topomaps.\n"
        "- BNCI: no local `artifacts/cache/bnci_*` found; will use motor-channel values from "
        "`erd_ers_trial_level.csv` with MNE interpolation (documented in `fig_topoplots_notes.md`).\n"
        "- No baseline-corrected ERD/ERS; label as mu-band power topography.\n"
    )
    lines.append("\n## Statistical ranking\n\n")
    lines.append(
        "- Friedman tests available (`model_friedman_results.csv`).\n"
        "- CD diagram attempted for PhysioNet EA=True (k=5, n=10 repetitions).\n"
        "- Holm post-hoc heatmap generated as complementary/alternative figure.\n"
    )
    return "".join(lines)
