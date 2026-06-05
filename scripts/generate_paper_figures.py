#!/usr/bin/env python3
"""Generate paper-ready figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.figures import plot_repeated_holdout_accuracy, write_figure_captions
from physionet_mi.paths import paper_figures_dir, publishable_runs_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    results_root = args.results_root or publishable_runs_dir(ROOT)
    output_dir = args.output_dir or paper_figures_dir(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    for summary in results_root.glob("**/repeated_holdout_summary.csv"):
        plot_repeated_holdout_accuracy(summary, output_dir)
    write_figure_captions(output_dir)
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
