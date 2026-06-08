#!/usr/bin/env python3
"""Generate paper-ready figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.figures import generate_all_paper_figures
from physionet_mi.paths import paper_figures_dir, publishable_runs_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    results_root = args.results_root or publishable_runs_dir(ROOT)
    output_dir = args.output_dir or paper_figures_dir(ROOT)

    paths = generate_all_paper_figures(results_root, output_dir)
    print(f"Figures written to {output_dir}")
    for name, path in paths.items():
        print(f"  {name}: {path or 'skipped'}")


if __name__ == "__main__":
    main()
