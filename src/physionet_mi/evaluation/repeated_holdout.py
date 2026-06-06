"""Repeated subject-disjoint hold-out evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from physionet_mi.config import load_config
from physionet_mi.data.cache import build_arrays_for_subject_split, load_raw_subject_dict
from physionet_mi.data.subject_dict import split_subjects_holdout
from physionet_mi.evaluation.bootstrap import aggregate_repeated_results
from physionet_mi.evaluation.config_matrix import resolve_config_path
from physionet_mi.evaluation.parallel_runner import (
    ModelJob,
    build_split_metadata,
    effective_inner_n_jobs,
    load_skipped_metrics,
    run_model_jobs,
)
from physionet_mi.evaluation.random_seeds import RepeatSeed, save_repeat_seeds
from physionet_mi.evaluation.runners import ALL_MODELS
from physionet_mi.evaluation.subject_splits import assert_no_subject_overlap

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


def _run_dir_tag(
    model_name: str,
    preprocess: str,
    repeat: RepeatSeed,
    *,
    legacy_explicit_seeds: bool,
) -> str:
    if legacy_explicit_seeds:
        return f"{model_name}_{preprocess}_seed{repeat.split_seed}"
    return f"{model_name}_{preprocess}_repeat{repeat.repeat_id}"


def run_repeated_holdout(
    *,
    repeat_seeds: list[RepeatSeed],
    models: Iterable[str],
    ea_modes: Iterable[bool],
    output_dir: Path,
    project_root: Path,
    dataset: str,
    test_size: float = 0.20,
    val_ratio: float = 0.15,
    skip_existing: bool = False,
    parallel_jobs: int = 1,
    master_seed: int | None = None,
    n_repeats: int | None = None,
    legacy_explicit_seeds: bool = False,
) -> pd.DataFrame:
    """Run repeated hold-out; returns results DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parallel_jobs = max(1, int(parallel_jobs))
    inner_n_jobs = effective_inner_n_jobs(parallel_jobs)
    rows: list[dict] = []
    failed: list[dict] = []
    model_list = list(models)

    save_repeat_seeds(
        repeat_seeds,
        output_dir / "repeat_seeds.json",
        master_seed=master_seed,
        n_repeats=n_repeats or len(repeat_seeds),
        legacy_explicit_seeds=legacy_explicit_seeds,
    )

    for use_ea in ea_modes:
        preprocess = "ea" if use_ea else "no_ea"
        cfg_path = resolve_config_path(dataset, preprocess, project_root)
        cfg = load_config(cfg_path, project_root=project_root)
        cfg.split.test_size = test_size
        cfg.split.val_ratio = val_ratio

        subj_data, ch_names, _ = load_raw_subject_dict(cfg)

        for repeat in repeat_seeds:
            split_seed = repeat.split_seed
            model_seed = repeat.model_seed
            cfg.split.random_state = split_seed
            cfg.train.seed = model_seed

            dev_ids, test_ids = split_subjects_holdout(subj_data, test_size, split_seed)
            assert_no_subject_overlap(dev_ids, np.array([], dtype=int), test_ids)

            arrays = build_arrays_for_subject_split(cfg, subj_data, dev_ids, test_ids, ch_names)
            n_train_subj = len(dev_ids)
            n_test_subj = len(test_ids)

            split_meta = build_split_metadata(
                dataset=dataset,
                seed=split_seed,
                train_subjects=dev_ids,
                test_subjects=test_ids,
                arrays=arrays,
                val_subjects=np.array([], dtype=int),
            )
            split_meta.update({
                "master_seed": master_seed,
                "n_repeats": n_repeats or len(repeat_seeds),
                "repeat_id": repeat.repeat_id,
                "split_seed": split_seed,
                "model_seed": model_seed,
                "legacy_explicit_seeds": legacy_explicit_seeds,
            })

            pending_jobs: list[ModelJob] = []
            for model_name in model_list:
                if model_name not in ALL_MODELS:
                    logger.warning("Skipping unknown model %s", model_name)
                    continue

                run_tag = _run_dir_tag(
                    model_name, preprocess, repeat, legacy_explicit_seeds=legacy_explicit_seeds
                )
                run_dir = output_dir / run_tag

                if skip_existing and _should_skip_existing_run(run_dir):
                    logger.info("Skipping existing %s", run_tag)
                    metrics = load_skipped_metrics(run_dir)
                    row = _row_from_repeat(
                        dataset=dataset,
                        repeat=repeat,
                        model_name=model_name,
                        use_ea=use_ea,
                        arrays=arrays,
                        n_train_subj=n_train_subj,
                        n_test_subj=n_test_subj,
                        metrics=metrics,
                        master_seed=master_seed,
                        n_repeats=n_repeats or len(repeat_seeds),
                    )
                    rows.append(row)
                    if row.get("status") != "ok":
                        failed.append(row)
                    continue

                pending_jobs.append(
                    ModelJob(
                        model_name=model_name,
                        cfg_path=str(cfg_path),
                        project_root=str(project_root),
                        run_dir=str(run_dir),
                        protocol="repeated_holdout",
                        inner_n_jobs=inner_n_jobs,
                        split_seed=split_seed,
                        model_seed=model_seed,
                        meta=split_meta,
                        arrays=arrays,
                    )
                )

            def _build_row(job: ModelJob, metrics: dict) -> dict:
                return _row_from_repeat(
                    dataset=dataset,
                    repeat=repeat,
                    model_name=job.model_name,
                    use_ea=use_ea,
                    arrays=arrays,
                    n_train_subj=n_train_subj,
                    n_test_subj=n_test_subj,
                    metrics=metrics,
                    master_seed=master_seed,
                    n_repeats=n_repeats or len(repeat_seeds),
                )

            batch_rows = run_model_jobs(
                pending_jobs,
                parallel_jobs=parallel_jobs,
                row_builder=_build_row,
            )
            rows.extend(batch_rows)
            failed.extend(r for r in batch_rows if r.get("status") != "ok")

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


def _row_from_repeat(
    *,
    dataset: str,
    repeat: RepeatSeed,
    model_name: str,
    use_ea: bool,
    arrays: dict,
    n_train_subj: int,
    n_test_subj: int,
    metrics: dict,
    master_seed: int | None,
    n_repeats: int,
) -> dict:
    return {
        "dataset": dataset,
        "master_seed": master_seed,
        "n_repeats": int(n_repeats),
        "repeat_id": int(repeat.repeat_id),
        "split_seed": int(repeat.split_seed),
        "model_seed": int(repeat.model_seed),
        "seed": int(repeat.split_seed),
        "model": model_name,
        "use_ea": bool(use_ea),
        "n_train_subjects": int(n_train_subj),
        "n_val_subjects": 0,
        "n_test_subjects": int(n_test_subj),
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
