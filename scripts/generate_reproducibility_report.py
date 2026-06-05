#!/usr/bin/env python3
"""Generate reproducibility report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paper.report_generator import generate_reproducibility_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=ROOT / "outputs_publishable")
    p.add_argument("--output", type=Path, default=ROOT / "outputs_publishable" / "reports" / "reproducibility_report.md")
    args = p.parse_args()
    generate_reproducibility_report(args.results_root, args.output, ROOT)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
