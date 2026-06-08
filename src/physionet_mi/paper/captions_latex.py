"""Figure captions and LaTeX snippets for IEEE manuscript."""

from __future__ import annotations

from pathlib import Path


CAPTIONS: list[dict[str, str]] = [
    {
        "file": "fig_pipeline_architecture",
        "short": "Benchmark pipeline from datasets to evaluation and auxiliary analyses.",
        "long": (
            "End-to-end workflow for the EEG motor-imagery benchmark. "
            "PhysioNet MI and BNCI2014-001 undergo shared preprocessing with an optional "
            "Euclidean Alignment (EA) ablation. Five decoders (CSP+SVM, FBCSP+LDA, Riemannian MDM, "
            "Riemannian tangent-space logistic regression, and EEGNet) are evaluated under "
            "repeated subject-disjoint hold-out (master\\_seed=42, ten repetitions). "
            "Subjects never appear in both training and test partitions."
        ),
        "interpretation": "Establishes strict subject-disjoint evaluation and where EA enters the pipeline.",
        "section": "Methods",
    },
    {
        "file": "fig_topoplots_motor_imagery",
        "short": "Mu-band spatial power maps for left- and right-hand imagery.",
        "long": (
            "Mu-band (8--13~Hz) power topography averaged across trials of each imagined class. "
            "Maps are \\textbf{not} baseline-corrected ERD/ERS. "
            "PhysioNet uses full 64-channel montage; BNCI uses motor channels with head interpolation "
            "when full cache is unavailable."
        ),
        "interpretation": "Secondary physiological context; contralateral motor patterns are descriptive only.",
        "section": "Discussion / Supplementary",
    },
    {
        "file": "fig_topoplots_before_after_ea",
        "short": "PhysioNet mu-band topography before and after EA.",
        "long": (
            "Mu-band power topography on PhysioNet before and after Euclidean Alignment "
            "for left- and right-hand motor imagery trials."
        ),
        "interpretation": "Visualizes spatial redistribution of band power after covariance alignment.",
        "section": "Discussion",
    },
    {
        "file": "fig_repeated_holdout_accuracy",
        "short": "Mean $\\pm$ SD balanced accuracy across ten repeated hold-outs.",
        "long": (
            "(A) PhysioNet MI and (B) BNCI2014-001. "
            "Balanced accuracy (mean $\\pm$ standard deviation over ten subject-disjoint repetitions) "
            "for each model with and without EA."
        ),
        "interpretation": "Primary quantitative comparison; CSP+SVM leads on PhysioNet, EEGNet on BNCI.",
        "section": "Results",
    },
    {
        "file": "fig_repeated_holdout_distributions",
        "short": "Per-repetition balanced accuracy distributions.",
        "long": (
            "Violin plots of balanced accuracy for each model and EA condition across ten "
            "repeated hold-out partitions, with jittered per-repetition points."
        ),
        "interpretation": "Shows repetition-level variability beyond mean$\\pm$SD bars.",
        "section": "Results",
    },
    {
        "file": "fig_ea_gain",
        "short": "EA-induced change in balanced accuracy by model.",
        "long": "Delta balanced accuracy (EA minus no-EA) for each model on PhysioNet and BNCI2014-001.",
        "interpretation": "Highlights EA benefit for covariance-based pipelines.",
        "section": "Results",
    },
    {
        "file": "fig_critical_difference",
        "short": "Nemenyi critical difference diagram (PhysioNet, EA).",
        "long": (
            "Average-rank critical difference diagram for the five models on PhysioNet "
            "with EA enabled (ten repeated hold-out blocks)."
        ),
        "interpretation": "Models connected by horizontal bars are not significantly different at $\\alpha{=}0.05$.",
        "section": "Results",
    },
    {
        "file": "fig_model_ranking_significance",
        "short": "Holm-adjusted pairwise significance heatmaps.",
        "long": (
            "Pairwise Wilcoxon post-hoc $p$-values (Holm corrected) between models for each dataset "
            "and EA condition. Asterisks indicate $p<0.05$."
        ),
        "interpretation": "Complements Friedman omnibus tests with explicit pairwise comparisons.",
        "section": "Results",
    },
    {
        "file": "fig_subject_level_variability",
        "short": "Inter-subject accuracy variability.",
        "long": "Distribution of per-subject mean accuracy aggregated across model runs.",
        "interpretation": "Supports discussion of heterogeneous BCI user performance.",
        "section": "Discussion",
    },
    {
        "file": "fig_ea_covariance_diagnostics",
        "short": "Covariance dispersion before and after EA.",
        "long": (
            "Mean Frobenius distance of trial covariance matrices within and between subjects "
            "before and after Euclidean Alignment."
        ),
        "interpretation": "Explains why EA disproportionately helps CSP/FBCSP/Riemannian models.",
        "section": "Methods / Discussion",
    },
    {
        "file": "fig_lateralization_by_class",
        "short": "Mu-band lateralization distributions by imagined class.",
        "long": (
            "Trial-level mu-band lateralization index distributions for left- vs right-hand imagery. "
            "Not baseline-corrected ERD/ERS."
        ),
        "interpretation": "Secondary neurophysiological analysis; not primary performance evidence.",
        "section": "Discussion / Supplementary",
    },
]


def write_figure_captions_md(path: Path) -> None:
    lines = ["# Figure captions\n\n"]
    lines.append("| File | Section | Short caption |\n| --- | --- | --- |\n")
    for c in CAPTIONS:
        lines.append(f"| `{c['file']}` | {c['section']} | {c['short']} |\n")
    lines.append("\n## Detailed captions\n\n")
    for c in CAPTIONS:
        lines.append(f"### `{c['file']}`\n\n")
        lines.append(f"**Long caption:** {c['long']}\n\n")
        lines.append(f"**Interpretation:** {c['interpretation']}\n\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_latex_snippets(path: Path) -> None:
    snippets = [
        ("fig_pipeline_architecture", "figure*", "0.92"),
        ("fig_repeated_holdout_accuracy", "figure*", "0.95"),
        ("fig_repeated_holdout_distributions", "figure*", "0.95"),
        ("fig_ea_gain", "figure", "0.48"),
        ("fig_model_ranking_significance", "figure*", "0.92"),
        ("fig_topoplots_motor_imagery", "figure*", "0.92"),
        ("fig_ea_covariance_diagnostics", "figure", "0.48"),
        ("fig_subject_level_variability", "figure*", "0.92"),
    ]
    lines = ["% Auto-generated LaTeX figure snippets (IEEE)\n", "% \\graphicspath{{artifacts/paper/figures/}}\n\n"]
    for stem, env, width in snippets:
        cap = next((c for c in CAPTIONS if c["file"] == stem), None)
        if not cap:
            continue
        label = stem.replace("fig_", "fig:")
        lines.append(f"\\begin{{{env}}}[t]\n")
        lines.append("  \\centering\n")
        lines.append(f"  \\includegraphics[width={width}\\textwidth]{{artifacts/paper/figures/{stem}.pdf}}\n")
        lines.append(f"  \\caption{{{cap['long']}}}\n")
        lines.append(f"  \\label{{{label}}}\n")
        lines.append(f"\\end{{{env}}}\n\n")
    path.write_text("".join(lines), encoding="utf-8")
