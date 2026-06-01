"""Leave-One-Subject-Out cross-validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from physionet_mi.config import ExperimentConfig, load_config
from physionet_mi.constants import class_to_workshop_label
from physionet_mi.data.cache import load_full_cohort_arrays
from physionet_mi.data.subject_dict import (
    loso_test_subjects,
    mask_by_subjects,
    split_val_subjects_stratified,
    validate_no_subject_overlap,
)
from physionet_mi.datasets.eeg_dataset import EEGDataset, make_dataloader
from physionet_mi.evaluation.metrics import compute_metrics, predict_loader
from physionet_mi.evaluation.reporting import save_run_artifacts
from physionet_mi.models.registry import build_model, default_run_prefix
from physionet_mi.training.trainer import fit
from physionet_mi.utils.device import get_device
from physionet_mi.utils.logging import setup_logging
from physionet_mi.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def run_loso(
    cfg: ExperimentConfig,
    run_name: str = "loso",
    max_folds: int | None = None,
) -> dict:
    seed_everything(cfg.train.seed)
    data = load_full_cohort_arrays(cfg)
    X, y, groups = data["X"], data["y"], data["groups"]

    test_subjects = loso_test_subjects(groups)
    if max_folds is not None:
        test_subjects = test_subjects[:max_folds]

    out_root = cfg.outputs_path() / run_name
    fold_metrics: list[dict] = []

    for fold_idx, test_sid in enumerate(test_subjects):
        logger.info("LOSO fold %d/%d test_subject=%s", fold_idx + 1, len(test_subjects), test_sid)
        test_mask = groups == test_sid
        train_val_mask = ~test_mask

        X_tv, y_tv, g_tv = X[train_val_mask], y[train_val_mask], groups[train_val_mask]
        X_te, y_te, g_te = X[test_mask], y[test_mask], groups[test_mask]

        y_workshop = np.array([class_to_workshop_label(int(c)) for c in y_tv])
        train_subj, val_subj = split_val_subjects_stratified(
            g_tv, y_workshop, cfg.split.val_ratio, cfg.split.random_state + fold_idx
        )
        tr_mask = mask_by_subjects(g_tv, train_subj)
        va_mask = mask_by_subjects(g_tv, val_subj)
        validate_no_subject_overlap(g_tv[tr_mask], g_tv[va_mask])

        train_ds = EEGDataset(X_tv[tr_mask], y_tv[tr_mask], g_tv[tr_mask], cfg.train.trial_wise_normalize)
        val_ds = EEGDataset(X_tv[va_mask], y_tv[va_mask], g_tv[va_mask], cfg.train.trial_wise_normalize)
        test_ds = EEGDataset(X_te, y_te, g_te, cfg.train.trial_wise_normalize)

        train_loader = make_dataloader(train_ds, cfg.train.batch_size, True)
        val_loader = make_dataloader(val_ds, cfg.train.batch_size, False) if len(val_ds) > 0 else None
        test_loader = make_dataloader(test_ds, cfg.train.batch_size, False)

        fold_dir = out_root / f"fold_{test_sid}"
        model = build_model(cfg)
        result = fit(model, train_loader, val_loader, cfg, fold_dir)

        device = get_device(cfg.train.device)
        model.to(device)
        y_pred, y_true, _ = predict_loader(model, test_loader, device)
        m = compute_metrics(y_true, y_pred)
        m["test_subject"] = int(test_sid)
        m["fold"] = fold_idx
        fold_metrics.append(m)
        save_run_artifacts(
            fold_dir,
            y_true,
            y_pred,
            {"fold": fold_idx, "test_subject": int(test_sid)},
            history=result.history,
        )

    df = pd.DataFrame(fold_metrics)
    summary = {
        "protocol": "loso",
        "use_ea": cfg.preprocess.use_ea,
        "dataset": cfg.data.dataset,
        "model": cfg.model.name,
        "n_folds": len(fold_metrics),
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std()),
        "balanced_accuracy_mean": float(df["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(df["balanced_accuracy"].std()),
        "macro_f1_mean": float(df["macro_f1"].mean()),
        "macro_f1_std": float(df["macro_f1"].std()),
        "kappa_mean": float(df["kappa"].mean()),
        "kappa_std": float(df["kappa"].std()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "fold_metrics.csv", index=False)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("LOSO summary: %s", summary)
    return summary


def main(config_path: str, run_name: str | None = None, max_folds: int | None = None, project_root: Path | None = None) -> None:
    setup_logging()
    config_path_p = Path(config_path).resolve()
    root = project_root or config_path_p.parent.parent
    cfg = load_config(config_path_p, project_root=root)
    name = run_name or f"{default_run_prefix(cfg)}_loso"
    run_loso(cfg, name, max_folds=max_folds)
