"""Rebuild repeated hold-out aggregate CSVs from per-run folders (no retraining)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from physionet_mi.evaluation.bootstrap import aggregate_repeated_results

logger = logging.getLogger(__name__)

RUN_DIR_PATTERN = re.compile(
    r"^(?P<model>.+)_(?P<preprocess>ea|no_ea)_(?P<tag>repeat\d+|seed\d+)$"
)


def _parse_run_dir_name(name: str) -> tuple[str, bool] | None:
    m = RUN_DIR_PATTERN.match(name)
    if not m:
        return None
    return m.group("model"), m.group("preprocess") == "ea"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _row_from_run_dir(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    meta_path = run_dir / "split_metadata.json"
    metrics = _read_json(metrics_path)
    meta = _read_json(meta_path) or {}

    if metrics is None:
        return None

    parsed = _parse_run_dir_name(run_dir.name)
    model = metrics.get("model") or (parsed[0] if parsed else None)
    if model is None:
        logger.warning("Could not determine model for %s", run_dir.name)
        return None

    if "use_ea" in metrics:
        use_ea = bool(metrics["use_ea"])
    elif parsed is not None:
        use_ea = parsed[1]
    elif "ea" in run_dir.name and "no_ea" not in run_dir.name:
        use_ea = True
    else:
        use_ea = False

    dataset = metrics.get("dataset") or meta.get("dataset", "")
    split_seed = meta.get("split_seed", meta.get("seed"))
    model_seed = meta.get("model_seed", split_seed)
    repeat_id = meta.get("repeat_id")
    if repeat_id is None and "repeat" in run_dir.name:
        try:
            repeat_id = int(run_dir.name.rsplit("repeat", 1)[-1])
        except ValueError:
            repeat_id = None

    train = meta.get("train", {})
    test = meta.get("test", {})
    status = metrics.get("status", "ok")
    if status != "error" and "accuracy" not in metrics:
        status = "error"
        metrics.setdefault("error_message", "missing accuracy in metrics.json")

    return {
        "dataset": dataset,
        "master_seed": meta.get("master_seed"),
        "n_repeats": meta.get("n_repeats"),
        "repeat_id": repeat_id,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "seed": split_seed,
        "model": model,
        "use_ea": use_ea,
        "n_train_subjects": train.get("n_subjects", np.nan),
        "n_val_subjects": 0,
        "n_test_subjects": test.get("n_subjects", np.nan),
        "n_train_trials": train.get("n_trials", np.nan),
        "n_val_trials": 0,
        "n_test_trials": test.get("n_trials", np.nan),
        "accuracy": metrics.get("accuracy", np.nan),
        "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
        "macro_f1": metrics.get("macro_f1", np.nan),
        "kappa": metrics.get("kappa", np.nan),
        "best_epoch": metrics.get("best_epoch"),
        "train_time_seconds": metrics.get("train_time_seconds", np.nan),
        "status": status,
        "error_message": metrics.get("error_message", ""),
        "run_dir": run_dir.name,
    }


def rebuild_repeated_holdout_artifacts(holdout_dir: Path) -> pd.DataFrame:
    """
    Scan run folders under *holdout_dir* and rewrite aggregate CSV/JSON files.

    Does not train models or modify per-run artifacts.
    """
    holdout_dir = holdout_dir.resolve()
    if not holdout_dir.is_dir():
        raise FileNotFoundError(f"Hold-out directory not found: {holdout_dir}")

    rows: list[dict] = []
    for child in sorted(holdout_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / "metrics.json").exists():
            continue
        row = _row_from_run_dir(child)
        if row is not None:
            rows.append(row)

    if not rows:
        logger.warning("No run folders with metrics.json found in %s", holdout_dir)
        empty = pd.DataFrame()
        empty.to_csv(holdout_dir / "repeated_holdout_results.csv", index=False)
        return empty

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ("model", "use_ea", "repeat_id", "run_dir") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    if "run_dir" in df.columns:
        df = df.drop(columns=["run_dir"])

    df.to_csv(holdout_dir / "repeated_holdout_results.csv", index=False)
    df.to_json(holdout_dir / "repeated_holdout_results.json", orient="records", indent=2)

    ok_df = df[df["status"] == "ok"] if "status" in df.columns else df
    failed_df = df[df["status"] != "ok"] if "status" in df.columns else pd.DataFrame()

    if len(ok_df):
        summary = aggregate_repeated_results(
            ok_df,
            group_cols=["dataset", "model", "use_ea"],
        )
        summary.to_csv(holdout_dir / "repeated_holdout_summary.csv", index=False)

        ea_pivot = ok_df.pivot_table(
            index=["dataset", "model"],
            columns="use_ea",
            values="balanced_accuracy",
            aggfunc="mean",
        )
        if True in ea_pivot.columns and False in ea_pivot.columns:
            ea_gain = ea_pivot[True] - ea_pivot[False]
            ea_gain.reset_index().rename(
                columns={True: "bal_acc_ea", False: "bal_acc_no_ea"}
            ).to_csv(holdout_dir / "ea_gain_summary.csv", index=False)

        ranking = (
            ok_df.groupby(["dataset", "use_ea"])["balanced_accuracy"]
            .mean()
            .reset_index()
        )
        ranking.to_csv(holdout_dir / "ranking_by_dataset.csv", index=False)

    failed_path = holdout_dir / "failed_runs.csv"
    if len(failed_df):
        failed_df.to_csv(failed_path, index=False)
    elif failed_path.exists():
        failed_path.unlink()

    logger.info(
        "Rebuilt %s: %d rows (%d ok, %d failed)",
        holdout_dir.name,
        len(df),
        len(ok_df),
        len(failed_df),
    )
    return df


def rebuild_all_holdout_roots(repeated_holdout_root: Path, datasets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Rebuild every dataset folder under repeated_holdout/."""
    repeated_holdout_root = repeated_holdout_root.resolve()
    out: dict[str, pd.DataFrame] = {}
    if datasets:
        names = datasets
    else:
        names = sorted(
            p.name for p in repeated_holdout_root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    for name in names:
        holdout_dir = repeated_holdout_root / name
        if holdout_dir.is_dir():
            out[name] = rebuild_repeated_holdout_artifacts(holdout_dir)
    return out
