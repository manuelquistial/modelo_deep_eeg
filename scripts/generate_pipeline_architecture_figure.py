#!/usr/bin/env python3
"""Regenerate fig_pipeline_architecture and refresh related manuscript metadata."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.captions_latex import write_figure_captions_md, write_latex_snippets
from physionet_mi.paper.pipeline_figure import generate_pipeline_architecture_figure, write_pipeline_notes
from physionet_mi.paths import artifacts_root, reports_dir


def _update_pipeline_report(report_path: Path, *, figures_dir: Path, art: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    png = figures_dir / "fig_pipeline_architecture.png"
    pdf = figures_dir / "fig_pipeline_architecture.pdf"
    lines = [
        "# Generated missing figures report\n\n",
        f"_Pipeline-only regeneration: {now}_\n\n",
        f"- Artifacts root: `{art}`\n",
        f"- Output dir: `{figures_dir}`\n\n",
        "## Generated in this run\n\n",
        "- fig_pipeline_architecture\n",
        "- figure_captions.md (pipeline entry refreshed)\n",
        "- latex_figure_snippets.tex (pipeline entry refreshed)\n",
        "- fig_pipeline_architecture_notes.md\n\n",
        "## Required file check\n\n",
        "| File | Exists |\n| --- | --- |\n",
        f"| `fig_pipeline_architecture.png` | {'yes' if png.exists() else '**no**'} |\n",
        f"| `fig_pipeline_architecture.pdf` | {'yes' if pdf.exists() else '**no**'} |\n",
        "\n## Notes\n\n",
        "- Only the pipeline architecture figure was regenerated in this run.\n",
        "- Training results and experiment outputs were not modified.\n",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    art = artifacts_root(ROOT)
    out = art / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    png, pdf = generate_pipeline_architecture_figure(out / "fig_pipeline_architecture")
    write_pipeline_notes(out / "fig_pipeline_architecture_notes.md")
    write_figure_captions_md(out / "figure_captions.md")
    write_latex_snippets(out / "latex_figure_snippets.tex")
    _update_pipeline_report(reports_dir(ROOT) / "generated_missing_figures_report.md", figures_dir=out, art=art)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Updated captions: {out / 'figure_captions.md'}")
    print(f"Updated LaTeX: {out / 'latex_figure_snippets.tex'}")
    print(f"Updated report: {reports_dir(ROOT) / 'generated_missing_figures_report.md'}")


if __name__ == "__main__":
    main()
