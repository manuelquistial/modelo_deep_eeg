#!/usr/bin/env python3
"""Repeated subject-disjoint hold-out evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.evaluation.random_seeds import resolve_repeat_seeds
from physionet_mi.evaluation.repeated_holdout import run_repeated_holdout
from physionet_mi.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="Repeated subject-disjoint hold-out")
    p.add_argument("--dataset", required=True, choices=["physionet", "bnci2014_001", "bnci"])
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Legacy: explicit split seed list. Prefer --master-seed and --n-repeats.",
    )
    p.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help="Master seed for generating repeated split/model seeds (default: 42)",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeated hold-out partitions (default: 10)",
    )
    p.add_argument(
        "--separate-model-seeds",
        action="store_true",
        help="Draw independent model seeds from the master RNG",
    )
    p.add_argument("--models", nargs="+", default=["fbcsp_lda", "csp_svm"])
    p.add_argument("--ea", default="both", help="true|false|both")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--only-classical", action="store_true")
    p.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help="Parallel workers for classical models per repetition (deep models stay on GPU)",
    )
    args = p.parse_args()

    try:
        repeat_records, master_seed, n_repeats, legacy = resolve_repeat_seeds(
            explicit_seeds=args.seeds,
            master_seed=args.master_seed,
            n_repeats=args.n_repeats,
            separate_model_seeds=args.separate_model_seeds,
        )
    except ValueError as exc:
        p.error(str(exc))

    if legacy:
        logger.warning(
            "Using explicit --seeds. For manuscript experiments, prefer --master-seed and --n-repeats."
        )
        print(
            "WARNING: Using explicit --seeds. For manuscript experiments, "
            "prefer --master-seed and --n-repeats.",
            flush=True,
        )
    else:
        logger.info(
            "Master-seed mode: master_seed=%s n_repeats=%s separate_model_seeds=%s",
            master_seed,
            n_repeats,
            args.separate_model_seeds,
        )
        print(
            f"Using master_seed={master_seed}, n_repeats={n_repeats} "
            f"(generated {len(repeat_records)} split seeds)",
            flush=True,
        )

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
        repeat_seeds=repeat_records,
        models=models,
        ea_modes=ea_modes,
        output_dir=args.output_dir,
        project_root=ROOT,
        test_size=args.test_size,
        skip_existing=args.skip_existing,
        parallel_jobs=args.parallel_jobs,
        master_seed=master_seed,
        n_repeats=n_repeats,
        legacy_explicit_seeds=legacy,
    )


if __name__ == "__main__":
    main()
