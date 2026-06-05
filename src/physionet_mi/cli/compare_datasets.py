"""Compare all pipelines across datasets and preprocess variants."""

import argparse
import json
from pathlib import Path

import pandas as pd

from physionet_mi.evaluation.compare import run_pipeline_comparison
from physionet_mi.evaluation.config_matrix import SUPPORTED_DATASETS, SUPPORTED_PREPROCESS
from physionet_mi.utils.logging import setup_logging


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty or "accuracy" not in df.columns:
        return
    hold = df[df["protocol"] == "holdout"]
    print("\nMean accuracy by dataset × preprocess × model (holdout):")
    grouped = hold.groupby(["dataset", "preprocess", "model"])["accuracy"].mean()
    print(grouped.sort_values(ascending=False))
    if (df["protocol"] == "loso").any():
        loso = df[df["protocol"] == "loso"]
        print("\nMean accuracy by dataset × preprocess × model (LOSO):")
        print(
            loso.groupby(["dataset", "preprocess", "model"])["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Run all models on each dataset and preprocess variant "
            "(shared deep_eeg cache: outlier removal, HPF, harmonize, optional EA)"
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="physionet,bnci2014_001",
        help=f"Comma-separated dataset keys ({', '.join(SUPPORTED_DATASETS)})",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="ea,no_ea",
        help=f"Comma-separated preprocess variants ({', '.join(SUPPORTED_PREPROCESS)})",
    )
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=3,
        help="LOSO folds per combo (ignored when --full-loso is set)",
    )
    parser.add_argument(
        "--full-loso",
        action="store_true",
        help="Run LOSO on all subjects (slow; PhysioNet ~108 folds per DL model)",
    )
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--skip-lda", action="store_true")
    parser.add_argument("--skip-csp-svm", action="store_true")
    parser.add_argument("--skip-eegnet", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/runs/baseline/pipeline_comparison.csv",
        help="CSV output path (JSON saved alongside with .json extension)",
    )
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    dataset_keys = _parse_csv_list(args.datasets)
    preprocess_keys = _parse_csv_list(args.preprocess)
    max_folds = None if args.full_loso else (None if args.skip_loso else args.max_folds)

    all_rows = run_pipeline_comparison(
        root,
        datasets=dataset_keys,
        preprocess_variants=preprocess_keys,
        max_folds=max_folds,
        skip_loso=args.skip_loso,
        skip_lda=args.skip_lda,
        skip_csp_svm=args.skip_csp_svm,
        skip_eegnet=args.skip_eegnet,
        force_cache=args.force_cache,
    )

    out_csv = root / args.output
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_csv.with_suffix(".json")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"Saved {out_csv}")
    _print_summary(df)


if __name__ == "__main__":
    main()
