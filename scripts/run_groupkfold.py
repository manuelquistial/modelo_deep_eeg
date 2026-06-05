#!/usr/bin/env python3
"""Subject-wise GroupKFold evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.evaluation.groupkfold import run_groupkfold
from physionet_mi.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="GroupKFold subject-wise evaluation")
    p.add_argument("--dataset", required=True, choices=["physionet", "bnci2014_001", "bnci"])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--models", nargs="+", default=["fbcsp_lda", "csp_svm"])
    p.add_argument("--ea", default="both")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--skip-deep", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    dataset = "bnci2014_001" if args.dataset == "bnci" else args.dataset
    ea_modes = [True, False] if args.ea == "both" else [args.ea.lower() in ("true", "1", "ea")]

    run_groupkfold(
        dataset=dataset,
        n_splits=args.n_splits,
        models=args.models,
        ea_modes=ea_modes,
        output_dir=args.output_dir,
        project_root=ROOT,
        skip_deep=args.skip_deep,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
