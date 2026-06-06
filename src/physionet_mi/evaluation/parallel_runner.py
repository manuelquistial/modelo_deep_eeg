"""Parallel execution helpers for publishable evaluation."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from physionet_mi.config import ExperimentConfig, load_config
from physionet_mi.evaluation.runners import DEEP_MODELS, evaluate_model_on_arrays
from physionet_mi.evaluation.subject_splits import save_split_metadata, summarize_split

logger = logging.getLogger(__name__)

CLASSICAL_MODELS = {m for m in ("fbcsp_lda", "csp_svm", "riemann_mdm", "riemann_ts_lr", "riemann_fgmdm", "riemann_ts_svm")}


def effective_inner_n_jobs(parallel_jobs: int) -> int:
    """Limit per-model sklearn parallelism when outer pool is active."""
    cpus = os.cpu_count() or 1
    if parallel_jobs <= 1:
        return -1
    return max(1, cpus // parallel_jobs)


def partition_models(models: Iterable[str]) -> tuple[list[str], list[str]]:
    classical: list[str] = []
    deep: list[str] = []
    for name in models:
        if name in DEEP_MODELS:
            deep.append(name)
        else:
            classical.append(name)
    return classical, deep


@dataclass(frozen=True)
class ModelJob:
    model_name: str
    cfg_path: str
    project_root: str
    run_dir: str
    protocol: str
    inner_n_jobs: int
    split_seed: int
    model_seed: int
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _sync_model_dims_from_arrays(cfg: ExperimentConfig, arrays: dict[str, np.ndarray]) -> None:
    """Match deep-model input dims to preprocessed arrays (config YAML may use defaults)."""
    X_dev = arrays.get("X_dev")
    if X_dev is not None and getattr(X_dev, "ndim", 0) == 3:
        cfg.model.n_channels = int(X_dev.shape[1])
        cfg.model.n_times = int(X_dev.shape[2])
        return
    meta = arrays.get("meta") or {}
    if "n_channels" in meta:
        cfg.model.n_channels = int(meta["n_channels"])
    if "model_n_times" in meta:
        cfg.model.n_times = int(meta["model_n_times"])


def _prepare_cfg(
    cfg_path: str,
    project_root: str,
    inner_n_jobs: int,
    split_seed: int,
    model_seed: int,
    arrays: dict[str, np.ndarray] | None = None,
) -> ExperimentConfig:
    cfg = load_config(cfg_path, project_root=Path(project_root))
    cfg.split.random_state = int(split_seed)
    cfg.train.seed = int(model_seed)
    cfg.csp_svm.seed = int(model_seed)
    if inner_n_jobs > 0:
        cfg.csp_svm.n_jobs = inner_n_jobs
    if arrays is not None:
        _sync_model_dims_from_arrays(cfg, arrays)
    return cfg


def _evaluate_job(job: ModelJob) -> dict[str, Any]:
    """Process-pool worker: train/evaluate one model on a fixed split."""
    run_dir = Path(job.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    split_path = run_dir / (
        "split_metadata.json" if job.protocol == "repeated_holdout" else "fold_metadata.json"
    )
    if job.meta:
        save_split_metadata(split_path, job.meta)

    cfg = _prepare_cfg(
        job.cfg_path,
        job.project_root,
        job.inner_n_jobs,
        job.split_seed,
        job.model_seed,
        job.arrays,
    )
    tag = f"{job.model_name} ({job.protocol})"
    logger.info("Starting %s -> %s", tag, run_dir.name)
    print(f"[parallel] start {run_dir.name}", flush=True)

    metrics = evaluate_model_on_arrays(
        job.model_name,
        cfg,
        job.arrays,
        run_dir,
        protocol=job.protocol,
    )
    print(
        f"[parallel] done {run_dir.name} status={metrics.get('status', 'ok')} "
        f"bal_acc={metrics.get('balanced_accuracy', 'n/a')}",
        flush=True,
    )
    return metrics


def run_model_jobs(
    jobs: list[ModelJob],
    *,
    parallel_jobs: int,
    row_builder: Callable[[ModelJob, dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run classical jobs in a process pool; deep jobs sequentially in parent."""
    if not jobs:
        return []

    classical_jobs = [j for j in jobs if j.model_name not in DEEP_MODELS]
    deep_jobs = [j for j in jobs if j.model_name in DEEP_MODELS]
    rows: list[dict[str, Any]] = []

    if classical_jobs and parallel_jobs > 1:
        logger.info(
            "Running %d classical job(s) with parallel_jobs=%d (inner n_jobs=%d)",
            len(classical_jobs),
            parallel_jobs,
            effective_inner_n_jobs(parallel_jobs),
        )
        with ProcessPoolExecutor(max_workers=parallel_jobs) as pool:
            futures = {pool.submit(_evaluate_job, job): job for job in classical_jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    metrics = fut.result()
                except Exception as exc:
                    logger.exception("Parallel job failed for %s", job.run_dir)
                    metrics = {
                        "status": "error",
                        "error_message": str(exc),
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "macro_f1": np.nan,
                        "kappa": np.nan,
                        "train_time_seconds": np.nan,
                        "best_epoch": None,
                    }
                rows.append(row_builder(job, metrics))
    else:
        for job in classical_jobs:
            metrics = _evaluate_job(job)
            rows.append(row_builder(job, metrics))

    for job in deep_jobs:
        logger.info("Running deep model sequentially on GPU: %s", job.model_name)
        metrics = _evaluate_job(job)
        rows.append(row_builder(job, metrics))

    return rows


def build_split_metadata(
    *,
    dataset: str,
    train_subjects: np.ndarray,
    test_subjects: np.ndarray,
    arrays: dict[str, np.ndarray],
    seed: int | None = None,
    fold: int | None = None,
    val_subjects: np.ndarray | None = None,
) -> dict[str, Any]:
    return summarize_split(
        dataset=dataset,
        seed=seed,
        fold=fold,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        y=np.concatenate([arrays["y_dev"], arrays["y_test"]]),
        groups=np.concatenate([arrays["groups_dev"], arrays["groups_test"]]),
    )


def load_skipped_metrics(run_dir: Path) -> dict[str, Any]:
    data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    data.setdefault("status", "ok")
    return data
