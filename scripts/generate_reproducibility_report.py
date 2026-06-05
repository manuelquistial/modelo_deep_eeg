#!/usr/bin/env python3
"""Generate reproducibility report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.report_generator import generate_reproducibility_report
from physionet_mi.paths import publishable_runs_dir, reports_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    results_root = args.results_root or publishable_runs_dir(ROOT)
    output = args.output or (reports_dir(ROOT) / "reproducibility_report.md")
    generate_reproducibility_report(results_root, output, ROOT)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
