#!/usr/bin/env python3
"""Generate paper-ready figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.figures import plot_repeated_holdout_accuracy, write_figure_captions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=ROOT / "outputs_publishable")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs_publishable" / "paper_figures")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for summary in args.results_root.glob("**/repeated_holdout_summary.csv"):
        plot_repeated_holdout_accuracy(summary, args.output_dir)
    write_figure_captions(args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
