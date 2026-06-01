"""Run all models on one dataset config and return metric rows.

All pipelines (EEGMe, EEGNet, FBCSP+LDA, CSP+SVM) share the same cached tensors
built by ``prepare_data`` / ``build_and_cache_holdout``: outlier removal, high-pass
filter, time harmonization, and optional Euclidean Alignment from ``preprocessing.py``.
"""

from __future__ import annotations

import copy
from pathlib import Path

from physionet_mi.baseline.csp_svm_classifier import run_csp_svm_holdout
from physionet_mi.baseline.lda_fbcsp import run_lda_holdout
from physionet_mi.config import ExperimentConfig, load_config
from physionet_mi.data.cache import build_and_cache_holdout
from physionet_mi.evaluation.config_matrix import iter_benchmark_configs
from physionet_mi.models.registry import default_run_prefix
from physionet_mi.training.holdout import run_holdout
from physionet_mi.training.loso import run_loso


def _row(
    dataset: str,
    model: str,
    protocol: str,
    use_ea: bool,
    run: str,
    metrics: dict,
    *,
    preprocess: str | None = None,
) -> dict:
    return {
        "dataset": dataset,
        "preprocess": preprocess or ("ea" if use_ea else "no_ea"),
        "model": model,
        "protocol": protocol,
        "use_ea": use_ea,
        "run": run,
        **metrics,
    }


def run_dataset_benchmark(
    cfg: ExperimentConfig,
    *,
    max_folds: int | None = 3,
    skip_loso: bool = False,
    skip_lda: bool = False,
    skip_csp_svm: bool = False,
    skip_eegnet: bool = False,
    force_cache: bool = False,
) -> list[dict]:
    """Prepare cache (if needed) and run all pipelines for one dataset."""
    build_and_cache_holdout(cfg, force=force_cache)
    rows: list[dict] = []
    ds = cfg.data.dataset
    ea = cfg.preprocess.use_ea

    dl_runs = [("eegme", cfg)]
    if not skip_eegnet:
        cfg_eegnet = load_config_from_same_root(cfg, model_name="eegnet")
        rows_cfg = cfg_eegnet
        dl_runs.append(("eegnet", rows_cfg))

    for model_name, model_cfg in dl_runs:
        model_cfg.model.name = model_name
        prefix = default_run_prefix(model_cfg)
        m = run_holdout(model_cfg, f"{prefix}_holdout")
        rows.append(_row(ds, model_name, "holdout", ea, f"{prefix}_holdout", m))
        if not skip_loso:
            s = run_loso(model_cfg, f"{prefix}_loso", max_folds=max_folds)
            rows.append(_row(
                ds, model_name, "loso", ea, f"{prefix}_loso",
                {
                    "accuracy": s["accuracy_mean"],
                    "balanced_accuracy": s["balanced_accuracy_mean"],
                    "macro_f1": s["macro_f1_mean"],
                    "kappa": s["kappa_mean"],
                    "accuracy_std": s.get("accuracy_std"),
                },
            ))

    if not skip_lda:
        m_lda = run_lda_holdout(cfg, f"lda_{'ea' if ea else 'no_ea'}_holdout")
        rows.append(_row(ds, "FBCSP+LDA", "holdout", ea, f"lda_{'ea' if ea else 'no_ea'}_holdout", m_lda))

    if not skip_csp_svm:
        m_csp = run_csp_svm_holdout(cfg, f"csp_svm_{'ea' if ea else 'no_ea'}_holdout")
        rows.append(_row(ds, "CSP+SVM", "holdout", ea, f"csp_svm_{'ea' if ea else 'no_ea'}_holdout", m_csp))

    return rows


def load_config_from_same_root(cfg: ExperimentConfig, model_name: str) -> ExperimentConfig:
    """Clone cfg with model.name set (same preprocess/data)."""
    new_cfg = copy.deepcopy(cfg)
    new_cfg.model.name = model_name
    if model_name == "eegnet":
        new_cfg.model.eegnet_dropout = 0.5
    return new_cfg


def benchmark_from_config_path(
    config_path: Path,
    project_root: Path,
    **kwargs,
) -> list[dict]:
    cfg = load_config(config_path, project_root=project_root)
    return run_dataset_benchmark(cfg, **kwargs)


def run_pipeline_comparison(
    project_root: Path,
    *,
    datasets: list[str] | None = None,
    preprocess_variants: list[str] | None = None,
    max_folds: int | None = 3,
    skip_loso: bool = False,
    skip_lda: bool = False,
    skip_csp_svm: bool = False,
    skip_eegnet: bool = False,
    force_cache: bool = False,
) -> list[dict]:
    """Benchmark all models across datasets × deep_eeg preprocess variants."""
    datasets = datasets or ["physionet", "bnci2014_001"]
    preprocess_variants = preprocess_variants or ["ea", "no_ea"]

    all_rows: list[dict] = []
    for dataset, preprocess, cfg_path in iter_benchmark_configs(
        datasets, preprocess_variants, project_root
    ):
        rows = benchmark_from_config_path(
            cfg_path,
            project_root,
            max_folds=max_folds,
            skip_loso=skip_loso,
            skip_lda=skip_lda,
            skip_csp_svm=skip_csp_svm,
            skip_eegnet=skip_eegnet,
            force_cache=force_cache,
        )
        for row in rows:
            row["dataset"] = dataset
            row["preprocess"] = preprocess
        all_rows.extend(rows)
    return all_rows
