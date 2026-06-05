#!/usr/bin/env python3
"""Statistical analysis on repeated hold-out results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.evaluation.bootstrap import aggregate_repeated_results
from physionet_mi.evaluation.statistical_tests import (
    friedman_by_dataset_ea,
    paired_wilcoxon_ea,
    pairwise_wilcoxon_models,
)
from physionet_mi.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.results.exists():
        (args.output_dir / "statistical_summary.md").write_text(
            "Insufficient evidence: run repeated hold-out first.\n", encoding="utf-8"
        )
        return

    df = pd.read_csv(args.results)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df

    bootstrap = aggregate_repeated_results(ok, group_cols=["dataset", "model", "use_ea"])
    bootstrap.to_csv(args.output_dir / "bootstrap_ci.csv", index=False)

    ea_w = paired_wilcoxon_ea(df)
    ea_w.to_csv(args.output_dir / "ea_wilcoxon_results.csv", index=False)

    friedman = friedman_by_dataset_ea(ok)
    friedman.to_csv(args.output_dir / "model_friedman_results.csv", index=False)

    posthoc = pairwise_wilcoxon_models(ok)
    posthoc.to_csv(args.output_dir / "model_pairwise_posthoc.csv", index=False)

    lines = ["# Statistical summary\n", f"- Input: `{args.results}`\n", f"- N rows: {len(df)}\n"]
    if len(ea_w):
        lines.append("\n## EA vs no-EA (Wilcoxon)\n")
        for _, r in ea_w.iterrows():
            lines.append(f"- {r['dataset']}/{r['model']}: p={r.get('p_value', 'NA')}\n")
    (args.output_dir / "statistical_summary.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
