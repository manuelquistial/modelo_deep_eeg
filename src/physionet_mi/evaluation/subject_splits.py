"""Reusable subject-disjoint split utilities for publishable evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from physionet_mi.data.subject_dict import (
    mask_by_subjects,
    split_subjects_holdout,
    split_val_subjects_stratified,
    validate_no_subject_overlap,
)


def labels_by_subject_from_groups(
    groups: np.ndarray,
    y: np.ndarray,
) -> dict[int, int]:
    """Majority class label per subject."""
    out: dict[int, int] = {}
    for sid in np.unique(groups):
        mask = groups == sid
        out[int(sid)] = int(pd.Series(y[mask]).mode().iloc[0])
    return out


def get_subject_disjoint_holdout_split(
    subject_ids: np.ndarray | list[int],
    labels_by_subject: dict[int, int],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subject hold-out split (wrapper around split_subjects_holdout)."""
    from physionet_mi.data.moabb_loader import SubjectRecord

    subj_data: dict[int, SubjectRecord] = {}
    for sid in subject_ids:
        sid = int(sid)
        label = labels_by_subject[sid]
        subj_data[sid] = {
            "trials": [np.zeros((10, 1), dtype=np.float32)],
            "labels": np.array([label], dtype=int),
            "run_ids": np.array([0], dtype=int),
        }
    return split_subjects_holdout(subj_data, test_size, random_state)


def split_dev_train_val_subjects(
    dev_subjects: np.ndarray,
    labels_by_subject: dict[int, int],
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split DEV subjects into TRAIN and VAL (for deep learning)."""
    groups = np.repeat(dev_subjects, [1] * len(dev_subjects))
    y = np.array([labels_by_subject[int(s)] for s in dev_subjects], dtype=int)
    return split_val_subjects_stratified(groups, y, val_ratio, random_state)


def make_groupkfold_splits(
    subject_ids: np.ndarray,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (test_subjects, dev_subjects) per fold."""
    subject_ids = np.asarray(subject_ids, dtype=int)
    if n_splits < 2 or n_splits > len(subject_ids):
        raise ValueError(f"n_splits={n_splits} invalid for {len(subject_ids)} subjects")
    gkf = GroupKFold(n_splits=n_splits)
    groups = subject_ids
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for dev_idx, test_idx in gkf.split(subject_ids, groups=groups):
        dev_subj = subject_ids[dev_idx]
        test_subj = subject_ids[test_idx]
        assert_no_subject_overlap(dev_subj, test_subj)
        folds.append((test_subj, dev_subj))
    return folds


def assert_no_subject_overlap(
    train_subjects: np.ndarray,
    val_subjects: np.ndarray,
    test_subjects: np.ndarray | None = None,
) -> None:
    """Fail loudly on any subject overlap across splits."""
    train_s = set(np.asarray(train_subjects, dtype=int).tolist())
    val_s = set(np.asarray(val_subjects, dtype=int).tolist())
    if train_s & val_s:
        raise ValueError(f"Train/val subject overlap: {train_s & val_s}")
    if test_subjects is not None:
        test_s = set(np.asarray(test_subjects, dtype=int).tolist())
        if train_s & test_s:
            raise ValueError(f"Train/test subject overlap: {train_s & test_s}")
        if val_s & test_s:
            raise ValueError(f"Val/test subject overlap: {val_s & test_s}")


def summarize_split(
    *,
    dataset: str,
    seed: int | None = None,
    fold: int | None = None,
    train_subjects: np.ndarray,
    val_subjects: np.ndarray | None,
    test_subjects: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Build split metadata dict for JSON export."""

    def _trial_stats(subj_arr: np.ndarray) -> dict[str, Any]:
        mask = mask_by_subjects(groups, subj_arr)
        yy = y[mask]
        uniq, cnt = np.unique(yy, return_counts=True)
        return {
            "n_subjects": int(len(subj_arr)),
            "n_trials": int(mask.sum()),
            "class_counts": {int(k): int(v) for k, v in zip(uniq, cnt)},
        }

    meta: dict[str, Any] = {
        "dataset": dataset,
        "train_subjects": [int(s) for s in np.asarray(train_subjects).tolist()],
        "test_subjects": [int(s) for s in np.asarray(test_subjects).tolist()],
        "train": _trial_stats(train_subjects),
        "test": _trial_stats(test_subjects),
    }
    if seed is not None:
        meta["seed"] = int(seed)
    if fold is not None:
        meta["fold"] = int(fold)
    if val_subjects is not None and len(val_subjects) > 0:
        meta["val_subjects"] = [int(s) for s in np.asarray(val_subjects).tolist()]
        meta["val"] = _trial_stats(val_subjects)
    return meta


def save_split_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
