#!/usr/bin/env python3
"""Generate LaTeX-ready paper tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.latex_tables import write_table_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=ROOT / "outputs_publishable")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs_publishable" / "paper_tables")
    args = p.parse_args()
    written = write_table_bundle(args.results_root, args.output_dir)
    print(f"Wrote {len(written)} tables to {args.output_dir}")


if __name__ == "__main__":
    main()
