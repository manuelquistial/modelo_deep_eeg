#!/usr/bin/env python3
"""Rebuild repeated hold-out CSVs from existing run folders (no retraining)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.evaluation.rebuild_holdout_csvs import (  # noqa: E402
    rebuild_all_holdout_roots,
    rebuild_repeated_holdout_artifacts,
)
from physionet_mi.paths import publishable_runs_dir  # noqa: E402
from physionet_mi.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(
        description=(
            "Rebuild repeated_holdout_results.csv and related summaries from "
            "per-run metrics.json folders. Does not retrain models."
        )
    )
    p.add_argument(
        "--holdout-dir",
        type=Path,
        default=None,
        help="Single dataset hold-out dir, e.g. artifacts/runs/publishable/repeated_holdout/bnci",
    )
    p.add_argument(
        "--repeated-holdout-root",
        type=Path,
        default=None,
        help="Parent repeated_holdout/ directory (rebuilds all or selected datasets)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset subfolders to rebuild (default: all under repeated-holdout-root)",
    )
    args = p.parse_args()

    if args.holdout_dir:
        df = rebuild_repeated_holdout_artifacts(args.holdout_dir)
        print(f"Rebuilt {args.holdout_dir}: {len(df)} rows")
        return

    root = args.repeated_holdout_root or (publishable_runs_dir(ROOT) / "repeated_holdout")
    results = rebuild_all_holdout_roots(root, datasets=args.datasets)
    for name, df in results.items():
        print(f"Rebuilt {root / name}: {len(df)} rows")
    if not results:
        print(f"No datasets rebuilt under {root}")


if __name__ == "__main__":
    main()
