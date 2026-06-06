"""Repeated subject-disjoint hold-out evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from physionet_mi.config import ExperimentConfig, load_config
from physionet_mi.data.cache import build_arrays_for_subject_split, load_raw_subject_dict
from physionet_mi.data.subject_dict import split_subjects_holdout
from physionet_mi.evaluation.bootstrap import aggregate_repeated_results
from physionet_mi.evaluation.config_matrix import resolve_config_path
from physionet_mi.evaluation.runners import ALL_MODELS, evaluate_model_on_arrays
from physionet_mi.evaluation.subject_splits import (
    assert_no_subject_overlap,
    save_split_metadata,
    summarize_split,
)

logger = logging.getLogger(__name__)


def _should_skip_existing_run(run_dir: Path) -> bool:
    """Skip only successful prior runs; retry failed or incomplete artifacts."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if data.get("status") == "error":
        return False
    return "accuracy" in data


def _parse_ea(ea: str | bool) -> list[bool]:
    if isinstance(ea, bool):
        return [ea]
    if ea == "both":
        return [True, False]
    if ea in ("true", "ea", "1"):
        return [True]
    return [False]


def run_repeated_holdout(
    *,
    dataset: str,
    seeds: Iterable[int],
    models: Iterable[str],
    ea_modes: Iterable[bool],
    output_dir: Path,
    project_root: Path,
    test_size: float = 0.20,
    val_ratio: float = 0.15,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Run repeated hold-out; returns results DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failed: list[dict] = []

    for use_ea in ea_modes:
        preprocess = "ea" if use_ea else "no_ea"
        cfg_path = resolve_config_path(dataset, preprocess, project_root)
        cfg = load_config(cfg_path, project_root=project_root)
        cfg.split.test_size = test_size
        cfg.split.val_ratio = val_ratio

        subj_data, ch_names, _ = load_raw_subject_dict(cfg)
        all_subjects = np.array(sorted(subj_data.keys()), dtype=int)

        for seed in seeds:
            cfg.split.random_state = int(seed)
            dev_ids, test_ids = split_subjects_holdout(subj_data, test_size, int(seed))
            assert_no_subject_overlap(dev_ids, np.array([], dtype=int), test_ids)

            arrays = build_arrays_for_subject_split(cfg, subj_data, dev_ids, test_ids, ch_names)
            n_train_subj = len(dev_ids)
            n_test_subj = len(test_ids)

            for model_name in models:
                if model_name not in ALL_MODELS:
                    logger.warning("Skipping unknown model %s", model_name)
                    continue
                run_tag = f"{model_name}_{preprocess}_seed{seed}"
                run_dir = output_dir / run_tag
                result_csv = run_dir / "metrics.json"

                if skip_existing and _should_skip_existing_run(run_dir):
                    logger.info("Skipping existing %s", run_tag)
                    metrics = json.loads(result_csv.read_text(encoding="utf-8"))
                    metrics.setdefault("status", "ok")
                    rows.append(_row_from_metrics(
                        dataset, seed, model_name, use_ea,
                        arrays, n_train_subj, 0, n_test_subj, metrics,
                    ))
                    continue

                split_meta = summarize_split(
                    dataset=dataset,
                    seed=seed,
                    train_subjects=dev_ids,
                    val_subjects=np.array([], dtype=int),
                    test_subjects=test_ids,
                    y=np.concatenate([arrays["y_dev"], arrays["y_test"]]),
                    groups=np.concatenate([arrays["groups_dev"], arrays["groups_test"]]),
                )
                save_split_metadata(run_dir / "split_metadata.json", split_meta)

                metrics = evaluate_model_on_arrays(
                    model_name, cfg, arrays, run_dir, protocol="repeated_holdout"
                )
                row = _row_from_metrics(
                    dataset, seed, model_name, use_ea,
                    arrays, n_train_subj, 0, n_test_subj, metrics,
                )
                rows.append(row)
                if metrics.get("status") != "ok":
                    failed.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "repeated_holdout_results.csv", index=False)
    df.to_json(output_dir / "repeated_holdout_results.json", orient="records", indent=2)

    if len(df):
        summary = aggregate_repeated_results(
            df[df["status"] == "ok"] if "status" in df else df,
            group_cols=["dataset", "model", "use_ea"],
        )
        summary.to_csv(output_dir / "repeated_holdout_summary.csv", index=False)

        ea_pivot = df.pivot_table(
            index=["dataset", "model"],
            columns="use_ea",
            values="balanced_accuracy",
            aggfunc="mean",
        )
        if True in ea_pivot.columns and False in ea_pivot.columns:
            ea_gain = ea_pivot[True] - ea_pivot[False]
            ea_gain.reset_index().rename(columns={True: "bal_acc_ea", False: "bal_acc_no_ea"}).to_csv(
                output_dir / "ea_gain_summary.csv", index=False
            )

        ranking = (
            df[df["status"] == "ok"]
            .groupby(["dataset", "use_ea"])["balanced_accuracy"]
            .mean()
            .reset_index()
        )
        ranking.to_csv(output_dir / "ranking_by_dataset.csv", index=False)

    if failed:
        pd.DataFrame(failed).to_csv(output_dir / "failed_runs.csv", index=False)

    return df


def _row_from_metrics(
    dataset, seed, model_name, use_ea, arrays, n_train, n_val, n_test, metrics,
) -> dict:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "model": model_name,
        "use_ea": bool(use_ea),
        "n_train_subjects": int(n_train),
        "n_val_subjects": int(n_val),
        "n_test_subjects": int(n_test),
        "n_train_trials": int(len(arrays["y_dev"])),
        "n_val_trials": 0,
        "n_test_trials": int(len(arrays["y_test"])),
        "accuracy": metrics.get("accuracy", np.nan),
        "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
        "macro_f1": metrics.get("macro_f1", np.nan),
        "kappa": metrics.get("kappa", np.nan),
        "best_epoch": metrics.get("best_epoch"),
        "train_time_seconds": metrics.get("train_time_seconds", np.nan),
        "status": metrics.get("status", "ok"),
        "error_message": metrics.get("error_message", ""),
    }
