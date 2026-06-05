#!/usr/bin/env python3
"""Generate LaTeX-ready paper tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.latex_tables import write_table_bundle
from physionet_mi.paths import paper_tables_dir, publishable_runs_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    results_root = args.results_root or publishable_runs_dir(ROOT)
    output_dir = args.output_dir or paper_tables_dir(ROOT)
    written = write_table_bundle(results_root, output_dir)
    print(f"Wrote {len(written)} tables to {output_dir}")


if __name__ == "__main__":
    main()
