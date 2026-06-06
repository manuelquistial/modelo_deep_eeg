#!/usr/bin/env python3
"""Generate final Markdown/JSON/CSV report from publishable experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.final_experiment_report import (  # noqa: E402
    auto_detect_results_root,
    generate_final_experiment_report,
)
from physionet_mi.paths import artifacts_root, baseline_runs_dir, reports_dir  # noqa: E402


def main() -> None:
    art = artifacts_root(ROOT)
    default_reports = reports_dir(ROOT)

    p = argparse.ArgumentParser(
        description=(
            "Inspect publishable experiment outputs and write final_experiment_report.md, "
            "final_experiment_summary.json, and artifact_index.csv. "
            "Reads only; does not modify experiment results."
        )
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Publishable results root. Default: auto-detect "
            "(prefers artifacts/, then outputs_publishable/)."
        ),
    )
    p.add_argument(
        "--legacy-results-root",
        type=Path,
        default=None,
        help="Fixed hold-out baseline root (default: artifacts/runs/baseline or outputs/).",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help=f"Markdown report path (default: {default_reports}/final_experiment_report.md).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=f"JSON summary path (default: {default_reports}/final_experiment_summary.json).",
    )
    p.add_argument(
        "--artifact-index",
        type=Path,
        default=None,
        help=f"Artifact index CSV (default: {default_reports}/artifact_index.csv).",
    )
    args = p.parse_args()

    results_root = args.results_root
    if results_root is None:
        results_root = auto_detect_results_root(ROOT)
    elif results_root.name == "outputs_publishable":
        pass
    elif results_root.name == "artifacts":
        pass

    legacy = args.legacy_results_root or baseline_runs_dir(ROOT)

    out_md = args.output_md or (default_reports / "final_experiment_report.md")
    out_json = args.output_json or (default_reports / "final_experiment_summary.json")
    out_index = args.artifact_index or (default_reports / "artifact_index.csv")

    paths = generate_final_experiment_report(
        ROOT,
        results_root=results_root,
        legacy_results_root=legacy,
        output_md=out_md,
        output_json=out_json,
        artifact_index=out_index,
    )

    print(f"Wrote Markdown report: {paths.output_md}")
    print(f"Wrote JSON summary:    {paths.output_json}")
    print(f"Wrote artifact index:  {paths.artifact_index}")
    print(f"Results root:          {paths.results_root}")


if __name__ == "__main__":
    main()
