#!/usr/bin/env python3
"""Bundle manuscript-ready figures, tables, text, and references into one folder."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.figures import generate_all_paper_figures  # noqa: E402
from physionet_mi.paper.references_bib import write_references_bib  # noqa: E402
from physionet_mi.paths import artifacts_root, paper_tables_dir, publishable_runs_dir  # noqa: E402

BUNDLE_NAME = "manuscript_bundle"

FIGURE_MANIFEST = [
    ("fig_repeated_holdout_accuracy", "Main results — balanced accuracy, panels A/B"),
    ("fig_ea_gain", "EA ablation — Δ balanced accuracy by model"),
    ("fig_subject_level_variability", "Inter-subject accuracy variability"),
    ("fig_ea_covariance_diagnostics", "EA covariance dispersion before/after"),
    ("fig_lateralization_by_class", "Mu-band lateralization by class (secondary)"),
]

COPY_FILES = [
    ("artifacts/paper/tables/generated_result_sentences.md", "text/generated_result_sentences.md"),
    ("artifacts/paper/tables/table_repeated_holdout_results.csv", "tables/table_repeated_holdout_results.csv"),
    ("artifacts/paper/tables/table_ea_gain_results.csv", "tables/table_ea_gain_results.csv"),
    ("artifacts/paper/tables/table_groupkfold_results.csv", "tables/table_groupkfold_results.csv"),
    ("literature_comparison/LITERATURE_COMPARISON.md", "references/LITERATURE_COMPARISON.md"),
    ("literature_comparison/comparison_data.json", "references/comparison_data.json"),
]


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_readme(bundle: Path, generated: dict[str, Path | None]) -> None:
    lines = [
        "# Manuscript asset bundle\n",
        "Single folder with figures, tables, LaTeX sentences, and references for the IEEE paper.\n\n",
        "## Figures (`figures/`)\n\n",
        "| File | Description |\n",
        "| --- | --- |\n",
    ]
    for stem, desc in FIGURE_MANIFEST:
        png = bundle / "figures" / f"{stem}.png"
        ok = "yes" if png.exists() else "missing"
        lines.append(f"| `{stem}.png/pdf` | {desc} ({ok}) |\n")

    lines.append("\n## Tables (`tables/`)\n")
    for _, rel in COPY_FILES:
        if rel.startswith("tables/"):
            p = bundle / rel
            lines.append(f"- `{rel}` — {'present' if p.exists() else 'missing'}\n")

    lines.append("\n## Text (`text/`)\n")
    lines.append("- `generated_result_sentences.md` — LaTeX-ready methodology and result bullets\n")
    lines.append("- `figure_captions.md` — IEEE figure captions\n")

    lines.append("\n## References (`references/`)\n")
    lines.append("- `references.bib` — BibTeX for `\\bibliography{references}` in IEEE LaTeX\n")
    lines.append("- `references_list.md` — human-readable key list\n")
    lines.append("- `LITERATURE_COMPARISON.md` / `comparison_data.json` — if bundled\n")

    lines.append("\n## Suggested LaTeX usage\n\n")
    lines.append("```latex\n")
    lines.append("\\begin{figure*}[t]\n")
    lines.append("  \\centering\n")
    lines.append("  \\includegraphics[width=\\textwidth]{figures/fig_repeated_holdout_accuracy.pdf}\n")
    lines.append("  \\caption{Repeated subject-disjoint hold-out balanced accuracy. (A) PhysioNet MI. (B) BNCI2014-001.}\n")
    lines.append("  \\label{fig:main_results}\n")
    lines.append("\\end{figure*}\n")
    lines.append("```\n")

    (bundle / "README.md").write_text("".join(lines), encoding="utf-8")


def _write_references_list(bundle: Path) -> None:
    keys = [
        ("hewu2020ea", "Euclidean Alignment — Methods"),
        ("lawhern2018eegnet", "EEGNet architecture"),
        ("ang2008fbcsp", "FBCSP"),
        ("ramoser2000csp", "CSP"),
        ("barachant2013riemann", "Riemannian MDM / geometry"),
        ("jayaram2018moabb", "MOABB datasets"),
        ("tangermann2012bciiv", "BNCI2014-001"),
        ("goldberger2000physionet", "PhysioNet"),
        ("altaheri2023atcnet", "Related work — BNCI deep MI"),
        ("pan2026dcascrcnet", "Related work — comparison table"),
        ("saibene2024review", "Related work — systematic review"),
    ]
    lines = ["# Key bibliography entries\n\n", "| BibTeX key | Use in paper |\n", "| --- | --- |\n"]
    for key, use in keys:
        lines.append(f"| `{key}` | {use} |\n")
    (bundle / "references" / "references_list.md").write_text("".join(lines), encoding="utf-8")


def bundle_manuscript_assets(
    project_root: Path,
    *,
    bundle_dir: Path | None = None,
    results_root: Path | None = None,
) -> Path:
    art = artifacts_root(project_root)
    pub = results_root or publishable_runs_dir(project_root)
    bundle = bundle_dir or (art / "paper" / BUNDLE_NAME)
    fig_dir = bundle / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_all_paper_figures(pub, fig_dir)

    for stem, _ in FIGURE_MANIFEST:
        for ext in (".png", ".pdf"):
            src = art / "paper" / "figures" / f"{stem}{ext}"
            dst = fig_dir / f"{stem}{ext}"
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)

    captions_src = art / "paper" / "figures" / "figure_captions.md"
    if captions_src.exists():
        (bundle / "text").mkdir(parents=True, exist_ok=True)
        shutil.copy2(captions_src, bundle / "text" / "figure_captions.md")

    for src_rel, dst_rel in COPY_FILES:
        _copy_if_exists(project_root / src_rel, bundle / dst_rel)

    write_references_bib(bundle / "references" / "references.bib")
    _write_references_list(bundle)
    _write_readme(bundle, generated)

    print(f"Bundle written to: {bundle}")
    for stem, path in generated.items():
        status = path.name if path else "not generated"
        print(f"  - {stem}: {status}")
    return bundle


def main() -> None:
    p = argparse.ArgumentParser(description="Bundle IEEE manuscript assets into artifacts/paper/manuscript_bundle/")
    p.add_argument("--bundle-dir", type=Path, default=None)
    p.add_argument("--results-root", type=Path, default=None)
    args = p.parse_args()
    bundle_manuscript_assets(ROOT, bundle_dir=args.bundle_dir, results_root=args.results_root)


if __name__ == "__main__":
    main()
