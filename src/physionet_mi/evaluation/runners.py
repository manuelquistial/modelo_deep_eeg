"""Model runners on pre-split arrays (reused by repeated hold-out / groupkfold)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import yaml

from physionet_mi.baselines.riemannian import run_riemannian_model
from physionet_mi.baseline.csp_svm_classifier import CSPSVMClassifier, _csp_svm_config_dict
from physionet_mi.baseline.lda_fbcsp import extract_fbcsp_features
from physionet_mi.config import ExperimentConfig
from physionet_mi.constants import class_to_workshop_label
from physionet_mi.data.subject_dict import mask_by_subjects, split_val_subjects_stratified
from physionet_mi.datasets.eeg_dataset import EEGDataset, make_dataloader
from physionet_mi.evaluation.metrics import compute_metrics, predict_loader
from physionet_mi.evaluation.reporting import save_run_artifacts
from physionet_mi.models.registry import build_model
from physionet_mi.training.trainer import fit
from physionet_mi.utils.device import get_device
from physionet_mi.utils.seed import seed_everything
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

logger = logging.getLogger(__name__)

CLASSICAL_MODELS = {"fbcsp_lda", "csp_svm", "riemann_mdm", "riemann_ts_lr", "riemann_fgmdm"}
DEEP_MODELS = {"eegnet", "eegme"}
ALL_MODELS = sorted(CLASSICAL_MODELS | DEEP_MODELS)


def _config_to_dict(cfg: ExperimentConfig) -> dict:
    return yaml.safe_load(yaml.safe_dump({
        "data": cfg.data.__dict__,
        "preprocess": cfg.preprocess.__dict__,
        "split": cfg.split.__dict__,
        "model": cfg.model.__dict__,
        "train": cfg.train.__dict__,
    }))


def run_fbcsp_lda_arrays(
    cfg: ExperimentConfig,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    X_dev_f, X_test_f = extract_fbcsp_features(X_dev, y_dev, X_test, cfg)
    y_dev_w = np.array([class_to_workshop_label(int(c)) for c in y_dev])
    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
        priors=[0.5, 0.5],
    )
    lda.fit(X_dev_f, y_dev_w)
    y_pred_w = lda.predict(X_test_f)
    from physionet_mi.constants import workshop_label_to_class

    y_pred = np.array([workshop_label_to_class(int(p)) for p in y_pred_w], dtype=int)
    return y_pred, time.perf_counter() - t0


def run_csp_svm_arrays(
    cfg: ExperimentConfig,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    clf = CSPSVMClassifier(_csp_svm_config_dict(cfg))
    clf.fit(X_dev, y_dev)
    y_pred = clf.predict(X_test)
    return y_pred, time.perf_counter() - t0


def run_deep_model_arrays(
    cfg: ExperimentConfig,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    groups_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    out_dir,
) -> tuple[np.ndarray, float, int]:
    seed_everything(cfg.train.seed)
    train_subj, val_subj = split_val_subjects_stratified(
        groups_dev, y_dev, cfg.split.val_ratio, cfg.split.random_state
    )
    train_mask = mask_by_subjects(groups_dev, train_subj)
    val_mask = mask_by_subjects(groups_dev, val_subj)

    train_ds = EEGDataset(
        X_dev[train_mask], y_dev[train_mask], groups_dev[train_mask],
        trial_wise_normalize=cfg.train.trial_wise_normalize,
    )
    val_ds = (
        EEGDataset(
            X_dev[val_mask], y_dev[val_mask], groups_dev[val_mask],
            trial_wise_normalize=cfg.train.trial_wise_normalize,
        )
        if len(val_subj) > 0
        else None
    )
    test_ds = EEGDataset(
        X_test, y_test, groups_test,
        trial_wise_normalize=cfg.train.trial_wise_normalize,
    )

    train_loader = make_dataloader(train_ds, cfg.train.batch_size, shuffle=True)
    val_loader = (
        make_dataloader(val_ds, cfg.train.batch_size, shuffle=False)
        if val_ds is not None
        else None
    )
    test_loader = make_dataloader(test_ds, cfg.train.batch_size, shuffle=False)

    t0 = time.perf_counter()
    model = build_model(cfg)
    result = fit(model, train_loader, val_loader, cfg, out_dir)
    device = get_device(cfg.train.device)
    model.to(device)
    y_pred, y_true, groups = predict_loader(model, test_loader, device)
    assert np.array_equal(y_true, y_test)
    return y_pred, time.perf_counter() - t0, result.best_epoch


def evaluate_model_on_arrays(
    model_name: str,
    cfg: ExperimentConfig,
    arrays: dict,
    out_dir,
    protocol: str = "repeated_holdout",
) -> dict[str, Any]:
    """Train/evaluate one model; return metrics row fields."""
    X_dev = arrays["X_dev"]
    y_dev = arrays["y_dev"]
    groups_dev = arrays["groups_dev"]
    X_test = arrays["X_test"]
    y_test = arrays["y_test"]
    groups_test = arrays["groups_test"]

    best_epoch = None
    try:
        if model_name == "fbcsp_lda":
            y_pred, train_time = run_fbcsp_lda_arrays(cfg, X_dev, y_dev, X_test, y_test)
        elif model_name == "csp_svm":
            y_pred, train_time = run_csp_svm_arrays(cfg, X_dev, y_dev, X_test, y_test)
        elif model_name.startswith("riemann"):
            y_pred, train_time, _ = run_riemannian_model(
                model_name, X_dev, y_dev, X_test, y_test
            )
        elif model_name in DEEP_MODELS:
            orig_name = cfg.model.name
            cfg.model.name = model_name
            y_pred, train_time, best_epoch = run_deep_model_arrays(
                cfg, X_dev, y_dev, groups_dev, X_test, y_test, groups_test, out_dir
            )
            cfg.model.name = orig_name
        else:
            raise ValueError(f"Unknown model: {model_name}")

        metrics = compute_metrics(y_test, y_pred)
        save_run_artifacts(
            out_dir,
            y_test,
            y_pred,
            _config_to_dict(cfg),
            extra={
                "protocol": protocol,
                "model": model_name,
                "use_ea": cfg.preprocess.use_ea,
                "dataset": cfg.data.dataset,
                "groups": groups_test,
                "best_epoch": best_epoch,
                "train_time_seconds": train_time,
                **metrics,
            },
        )
        row = {
            **metrics,
            "best_epoch": best_epoch,
            "train_time_seconds": train_time,
            "status": "ok",
            "error_message": "",
        }
        return row
    except Exception as e:
        logger.exception("Model %s failed: %s", model_name, e)
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "kappa": np.nan,
            "best_epoch": best_epoch,
            "train_time_seconds": np.nan,
            "status": "error",
            "error_message": str(e),
        }
