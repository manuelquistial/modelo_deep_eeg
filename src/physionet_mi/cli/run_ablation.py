"""Run full ablation: baselines + DL models × datasets × preprocess × holdout/loso."""

import argparse
import json
from pathlib import Path

import pandas as pd

from physionet_mi.evaluation.compare import run_pipeline_comparison
from physionet_mi.evaluation.config_matrix import SUPPORTED_DATASETS, SUPPORTED_PREPROCESS
from physionet_mi.utils.logging import setup_logging


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run ablation across models, datasets, and preprocess variants"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="physionet",
        help=f"Comma-separated datasets ({', '.join(SUPPORTED_DATASETS)})",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="ea,no_ea",
        help=f"Comma-separated preprocess variants ({', '.join(SUPPORTED_PREPROCESS)})",
    )
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=3, help="LOSO folds (ignored if --full-loso)")
    parser.add_argument(
        "--full-loso",
        action="store_true",
        help="LOSO on all subjects (slow)",
    )
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--skip-lda", action="store_true")
    parser.add_argument("--skip-csp-svm", action="store_true")
    parser.add_argument("--skip-eegnet", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    max_folds = None if args.full_loso else (None if args.skip_loso else args.max_folds)
    rows = run_pipeline_comparison(
        root,
        datasets=_parse_csv_list(args.datasets),
        preprocess_variants=_parse_csv_list(args.preprocess),
        max_folds=max_folds,
        skip_loso=args.skip_loso,
        skip_lda=args.skip_lda,
        skip_csp_svm=args.skip_csp_svm,
        skip_eegnet=args.skip_eegnet,
        force_cache=args.force_cache,
    )

    df = pd.DataFrame(rows)
    from physionet_mi.paths import baseline_runs_dir

    out = baseline_runs_dir(root) / "ablation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (baseline_runs_dir(root) / "ablation_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Saved ablation summary to {out}")
    if not df.empty and "accuracy" in df.columns:
        hold = df[df["protocol"] == "holdout"]
        print("\nMean accuracy by dataset × preprocess × model (holdout):")
        print(
            hold.groupby(["dataset", "preprocess", "model"])["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )


if __name__ == "__main__":
    main()
