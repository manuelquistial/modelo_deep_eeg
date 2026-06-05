#!/usr/bin/env python3
"""Subject-level analysis from prediction files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.analysis.subject_level import (
    collect_subject_level_metrics,
    plot_subject_level_boxplot,
    write_subject_level_summary,
)
from physionet_mi.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_subject_level_metrics(args.predictions_root)
    df.to_csv(args.output_dir / "subject_level_metrics.csv", index=False)
    if len(df):
        hardest = df.groupby("subject_id")["accuracy"].mean().sort_values().head(20)
        hardest.reset_index().to_csv(args.output_dir / "subject_error_ranking.csv", index=False)
        plot_subject_level_boxplot(df, args.output_dir / "subject_level_accuracy_boxplot.png")
    write_subject_level_summary(df, args.output_dir / "subject_level_summary.md")


if __name__ == "__main__":
    main()
