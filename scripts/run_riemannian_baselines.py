#!/usr/bin/env python3
"""Run Riemannian baselines via repeated hold-out (wrapper)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.evaluation.random_seeds import resolve_repeat_seeds


def main() -> None:
    p = argparse.ArgumentParser(description="Riemannian baselines (MDM, tangent-space LR)")
    p.add_argument("--dataset", required=True, choices=["physionet", "bnci2014_001", "bnci"])
    p.add_argument("--ea", default="both")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Legacy explicit split seeds. Prefer --master-seed and --n-repeats.",
    )
    p.add_argument("--master-seed", type=int, default=None)
    p.add_argument("--n-repeats", type=int, default=None)
    p.add_argument("--separate-model-seeds", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    try:
        _, master_seed, n_repeats, legacy = resolve_repeat_seeds(
            explicit_seeds=args.seeds,
            master_seed=args.master_seed,
            n_repeats=args.n_repeats,
            separate_model_seeds=args.separate_model_seeds,
        )
    except ValueError as exc:
        p.error(str(exc))

    ds = "bnci" if args.dataset == "bnci2014_001" else args.dataset
    holdout_dir = args.output_dir
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_repeated_holdout.py"),
        "--dataset",
        ds,
        "--models",
        "riemann_mdm",
        "riemann_ts_lr",
        "--ea",
        args.ea,
        "--output-dir",
        str(holdout_dir),
        "--only-classical",
    ]
    if legacy:
        cmd.extend(["--seeds", *[str(s) for s in args.seeds]])
    else:
        cmd.extend(["--master-seed", str(master_seed), "--n-repeats", str(n_repeats)])
    if args.separate_model_seeds:
        cmd.append("--separate-model-seeds")
    if args.skip_existing:
        cmd.append("--skip-existing")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

    results_csv = holdout_dir / "repeated_holdout_results.csv"
    if results_csv.exists():
        import pandas as pd

        df = pd.read_csv(results_csv)
        riemann_models = {"riemann_mdm", "riemann_ts_lr", "riemann_fgmdm", "riemann_ts_svm"}
        riemann_df = df[df["model"].isin(riemann_models)]
        out_csv = args.output_dir / "riemannian_results.csv"
        riemann_df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
