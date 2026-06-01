"""Hold-out 80/20 training and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from physionet_mi.config import ExperimentConfig, load_config
from physionet_mi.data.cache import load_holdout_arrays
from physionet_mi.data.subject_dict import (
    mask_by_subjects,
    split_val_subjects_stratified,
    validate_no_subject_overlap,
)
from physionet_mi.datasets.eeg_dataset import EEGDataset, make_dataloader
from physionet_mi.evaluation.metrics import compute_metrics, predict_loader
from physionet_mi.evaluation.reporting import save_run_artifacts
from physionet_mi.models.eegme import build_model
from physionet_mi.training.trainer import fit
from physionet_mi.utils.device import get_device
from physionet_mi.utils.logging import setup_logging
from physionet_mi.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def _config_to_dict(cfg: ExperimentConfig) -> dict:
    return yaml.safe_load(yaml.safe_dump({
        "data": cfg.data.__dict__,
        "preprocess": cfg.preprocess.__dict__,
        "split": cfg.split.__dict__,
        "model": cfg.model.__dict__,
        "train": cfg.train.__dict__,
        "eval": cfg.eval.__dict__,
    }))


def run_holdout(cfg: ExperimentConfig, run_name: str = "holdout") -> dict:
    seed_everything(cfg.train.seed)
    data = load_holdout_arrays(cfg)

    X_dev, y_dev, groups_dev = data["X_dev"], data["y_dev"], data["groups_dev"]
    X_test, y_test, groups_test = data["X_test"], data["y_test"], data["groups_test"]

    y_dev_workshop = y_dev  # already 0|1
    train_subj, val_subj = split_val_subjects_stratified(
        groups_dev,
        y_dev_workshop,
        cfg.split.val_ratio,
        cfg.split.random_state,
    )

    train_mask = mask_by_subjects(groups_dev, train_subj)
    val_mask = mask_by_subjects(groups_dev, val_subj)

    validate_no_subject_overlap(groups_dev[train_mask], groups_dev[val_mask])

    train_ds = EEGDataset(
        X_dev[train_mask], y_dev[train_mask], groups_dev[train_mask],
        trial_wise_normalize=cfg.train.trial_wise_normalize,
    )
    val_ds = EEGDataset(
        X_dev[val_mask], y_dev[val_mask], groups_dev[val_mask],
        trial_wise_normalize=cfg.train.trial_wise_normalize,
    )
    test_ds = EEGDataset(
        X_test, y_test, groups_test,
        trial_wise_normalize=cfg.train.trial_wise_normalize,
    )

    train_loader = make_dataloader(train_ds, cfg.train.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, cfg.train.batch_size, shuffle=False) if len(val_ds) > 0 else None
    test_loader = make_dataloader(test_ds, cfg.train.batch_size, shuffle=False)

    out_dir = cfg.outputs_path() / run_name
    model = build_model(cfg)
    result = fit(model, train_loader, val_loader, cfg, out_dir)

    device = get_device(cfg.train.device)
    model.to(device)
    y_pred, y_true, _ = predict_loader(model, test_loader, device)
    metrics = save_run_artifacts(
        out_dir,
        y_true,
        y_pred,
        _config_to_dict(cfg),
        history=result.history,
        extra={"protocol": "holdout", "use_ea": cfg.preprocess.use_ea, "best_epoch": result.best_epoch},
    )
    logger.info("Hold-out TEST metrics: %s", metrics)
    return metrics


def main(config_path: str, run_name: str | None = None, project_root: Path | None = None) -> None:
    setup_logging()
    config_path_p = Path(config_path).resolve()
    root = project_root or config_path_p.parent.parent
    cfg = load_config(config_path_p, project_root=root)
    name = run_name or f"dl_{'ea' if cfg.preprocess.use_ea else 'no_ea'}_holdout"
    run_holdout(cfg, name)
