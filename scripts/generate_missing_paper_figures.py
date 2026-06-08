#!/usr/bin/env python3
"""Generate missing IEEE manuscript figures from existing artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.captions_latex import write_figure_captions_md, write_latex_snippets
from physionet_mi.paper.distribution_figure import (
    generate_repeated_holdout_distributions,
    write_distribution_notes,
)
from physionet_mi.paper.figures import generate_all_paper_figures
from physionet_mi.paper.gap_analysis import audit_figures
from physionet_mi.paper.pipeline_figure import (
    generate_pipeline_architecture_figure,
    write_pipeline_notes,
)
from physionet_mi.paper.ranking_figure import (
    generate_statistical_ranking_figures,
    write_ranking_notes,
)
from physionet_mi.paper.topographic_figure import (
    generate_topographic_figures,
    write_topoplot_notes,
)
from physionet_mi.paths import artifacts_root, publishable_runs_dir, reports_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Generate missing paper figures from artifacts/")
    p.add_argument("--artifacts-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--gap-report", type=Path, default=None)
    args = p.parse_args()

    art = (args.artifacts_root or artifacts_root(ROOT)).resolve()
    out = (args.output_dir or (art / "paper" / "figures")).resolve()
    report_path = (args.report or (reports_dir(ROOT) / "generated_missing_figures_report.md")).resolve()
    gap_path = (args.gap_report or (reports_dir(ROOT) / "figure_gap_analysis.md")).resolve()
    pub = publishable_runs_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    gap_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: gap analysis (before generation)
    gap_path.write_text(audit_figures(out, pub), encoding="utf-8")
    print(f"Wrote gap analysis: {gap_path}")

    generated: list[str] = []
    skipped: list[str] = []

    # Existing supplementary figures (EA gain, etc.)
    supp = generate_all_paper_figures(pub, out)
    for name, path in supp.items():
        if path:
            generated.append(f"{name}: {path.name}")

    # Figure 1: pipeline
    generate_pipeline_architecture_figure(out / "fig_pipeline_architecture")
    write_pipeline_notes(out / "fig_pipeline_architecture_notes.md")
    generated.append("fig_pipeline_architecture")

    # Figure 2: topomaps
    topo = generate_topographic_figures(
        art,
        out / "fig_topoplots_motor_imagery",
        ea_stem=out / "fig_topoplots_before_after_ea",
    )
    write_topoplot_notes(
        out / "fig_topoplots_notes.md",
        bnci_sparse=not any((art / "cache").glob("bnci_*")),
    )
    if topo.get("motor_imagery"):
        generated.append("fig_topoplots_motor_imagery")
    else:
        skipped.append("fig_topoplots_motor_imagery")
    if topo.get("before_after_ea"):
        generated.append("fig_topoplots_before_after_ea")

    # Figure 3: distributions
    dist = generate_repeated_holdout_distributions(pub, out / "fig_repeated_holdout_distributions")
    write_distribution_notes(out / "fig_repeated_holdout_distributions_notes.md")
    if dist:
        generated.append("fig_repeated_holdout_distributions")

    # Figure 4: statistical ranking
    ranking = generate_statistical_ranking_figures(pub, out)
    write_ranking_notes(out / "fig_critical_difference_notes.md", ranking)
    if ranking.get("cd_generated"):
        generated.append("fig_critical_difference")
    if ranking.get("heatmap_paths"):
        generated.append("fig_model_ranking_significance")

    # Captions & LaTeX
    write_figure_captions_md(out / "figure_captions.md")
    write_latex_snippets(out / "latex_figure_snippets.tex")

    # Generation report
    now = datetime.now(timezone.utc).isoformat()
    req = [
        "fig_pipeline_architecture.png",
        "fig_topoplots_motor_imagery.png",
        "fig_repeated_holdout_distributions.png",
        "fig_critical_difference.png",
        "fig_model_ranking_significance.png",
    ]
    lines = [
        f"# Generated missing figures report\n\n",
        f"_Generated: {now}_\n\n",
        f"- Artifacts root: `{art}`\n",
        f"- Output dir: `{out}`\n",
        f"- Gap analysis: `{gap_path}`\n\n",
        "## Generated in this run\n\n",
    ]
    for g in generated:
        lines.append(f"- {g}\n")
    if skipped:
        lines.append("\n## Skipped / failed\n\n")
        for s in skipped:
            lines.append(f"- {s}\n")
    lines.append("\n## Required file check\n\n")
    lines.append("| File | Exists |\n| --- | --- |\n")
    for fname in req:
        ok = (out / fname).exists()
        lines.append(f"| `{fname}` | {'yes' if ok else '**no**'} |\n")
    lines.append("\n## Notes files\n\n")
    for note in sorted(out.glob("*_notes.md")):
        lines.append(f"- `{note.name}`\n")
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(f"Figures in: {out}")


if __name__ == "__main__":
    main()
