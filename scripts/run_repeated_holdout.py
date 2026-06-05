#!/usr/bin/env python3
"""Repeated subject-disjoint hold-out evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.evaluation.repeated_holdout import run_repeated_holdout
from physionet_mi.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="Repeated subject-disjoint hold-out")
    p.add_argument("--dataset", required=True, choices=["physionet", "bnci2014_001", "bnci"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--models", nargs="+", default=["fbcsp_lda", "csp_svm"])
    p.add_argument("--ea", default="both", help="true|false|both")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--only-classical", action="store_true")
    args = p.parse_args()

    dataset = "bnci2014_001" if args.dataset == "bnci" else args.dataset
    models = args.models
    if args.only_classical:
        models = [m for m in models if m in {"fbcsp_lda", "csp_svm", "riemann_mdm", "riemann_ts_lr", "riemann_fgmdm"}]

    if args.ea == "both":
        ea_modes = [True, False]
    else:
        ea_modes = [args.ea.lower() in ("true", "1", "ea")]

    run_repeated_holdout(
        dataset=dataset,
        seeds=args.seeds,
        models=models,
        ea_modes=ea_modes,
        output_dir=args.output_dir,
        project_root=ROOT,
        test_size=args.test_size,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
