"""Subject-level dict operations and splits."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from physionet_mi.constants import LABEL_ID_TO_NAME, workshop_label_to_class
from physionet_mi.data.moabb_loader import SubjectRecord

logger = logging.getLogger(__name__)


def subset_subject_dict(data_dict: dict[int, SubjectRecord], subject_ids: np.ndarray) -> dict[int, SubjectRecord]:
    return {int(sid): data_dict[int(sid)] for sid in subject_ids if int(sid) in data_dict}


def summarize_subject_dict(name: str, data_dict: dict[int, SubjectRecord]) -> None:
    if not data_dict:
        logger.info("[%s] empty", name)
        return
    all_labels = np.concatenate([s["labels"] for s in data_dict.values()])
    unique, counts = np.unique(all_labels, return_counts=True)
    shapes = [tuple(t.shape) for s in data_dict.values() for t in s["trials"]]
    logger.info(
        "[%s] subjects=%d trials=%d labels=%s common_shapes=%s",
        name,
        len(data_dict),
        sum(len(v["trials"]) for v in data_dict.values()),
        {LABEL_ID_TO_NAME.get(int(k), str(k)): int(v) for k, v in zip(unique, counts)},
        Counter(shapes).most_common(3),
    )


def flatten_subject_dict(
    data_dict: dict[int, SubjectRecord],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, groups, run_ids = [], [], [], []
    for sid, sdata in data_dict.items():
        for i, trial in enumerate(sdata["trials"]):
            X_list.append(trial)
            y_list.append(sdata["labels"][i])
            groups.append(sid)
            run_ids.append(sdata["run_ids"][i])
    return (
        X_list,
        np.asarray(y_list, dtype=int),
        np.asarray(groups, dtype=int),
        np.asarray(run_ids, dtype=int),
    )


def split_subjects_holdout(
    subj_data: dict[int, SubjectRecord],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_subjects = np.array(sorted(subj_data.keys()))
    subject_major_labels = []
    for sid in all_subjects:
        labels_sid = subj_data[sid]["labels"]
        subject_major_labels.append(pd.Series(labels_sid).mode().iloc[0])
    subject_major_labels = np.asarray(subject_major_labels)

    stratify = subject_major_labels if len(np.unique(subject_major_labels)) > 1 else None
    dev_subjects, test_subjects = train_test_split(
        all_subjects,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return dev_subjects, test_subjects


def split_val_subjects(
    train_groups: np.ndarray,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split unique train subjects into train/val (no trial leakage)."""
    unique_subjects = np.unique(train_groups)
    if len(unique_subjects) < 2:
        return unique_subjects, np.array([], dtype=int)

    major_labels = []
    for sid in unique_subjects:
        mask = train_groups == sid
        # groups don't carry labels here; use mode of placeholder - caller passes y
        major_labels.append(0)
    # stratify by subject index only if enough subjects
    n_val = max(1, int(round(len(unique_subjects) * val_ratio)))
    if n_val >= len(unique_subjects):
        n_val = len(unique_subjects) - 1

    train_subj, val_subj = train_test_split(
        unique_subjects,
        test_size=n_val,
        random_state=random_state,
    )
    return train_subj, val_subj


def split_val_subjects_stratified(
    train_groups: np.ndarray,
    train_y_workshop: np.ndarray,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split train subjects into train/val with stratification by majority class."""
    unique_subjects = np.unique(train_groups)
    if len(unique_subjects) < 2:
        return unique_subjects, np.array([], dtype=int)

    major_labels = []
    for sid in unique_subjects:
        mask = train_groups == sid
        labels = train_y_workshop[mask]
        major_labels.append(pd.Series(labels).mode().iloc[0])
    major_labels = np.asarray(major_labels)

    n_val = max(1, int(round(len(unique_subjects) * val_ratio)))
    n_classes = len(np.unique(major_labels))

    # Stratified split needs at least one subject per class in val.
    if len(np.unique(major_labels)) > 1 and n_val < n_classes:
        n_val = n_classes
    if n_val >= len(unique_subjects):
        n_val = max(1, len(unique_subjects) - 1)

    stratify = major_labels if len(np.unique(major_labels)) > 1 and n_val >= n_classes else None
    train_subj, val_subj = train_test_split(
        unique_subjects,
        test_size=n_val,
        random_state=random_state,
        stratify=stratify,
    )
    return train_subj, val_subj


def mask_by_subjects(groups: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    return np.isin(groups, subject_ids)


def trials_list_to_3d(
    X_list: list[np.ndarray],
    target_n_times: int | None = None,
    target_n_channels: int | None = None,
    crop_mode: str = "crop",
) -> np.ndarray:
    from physionet_mi.data.preprocessing import crop_or_pad_trial_time

    n_times_all = [trial.shape[0] for trial in X_list]
    if target_n_times is None:
        target_n_times = int(np.min(n_times_all))

    X_fixed = [crop_or_pad_trial_time(trial, target_n_times, mode=crop_mode).T for trial in X_list]
    X = np.stack(X_fixed, axis=0).astype(np.float32)

    if target_n_channels is not None and X.shape[1] != target_n_channels:
        raise ValueError(f"Expected {target_n_channels} channels, got {X.shape[1]}")
    return X


def to_model_arrays(
    data_dict: dict[int, SubjectRecord],
    target_n_times: int,
    n_channels: int,
    crop_mode: str = "crop",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert subject dict to (N, C, T), class labels 0|1, subject groups."""
    X_list, y_workshop, groups, _ = flatten_subject_dict(data_dict)
    X = trials_list_to_3d(
        X_list,
        target_n_times=target_n_times,
        target_n_channels=n_channels,
        crop_mode=crop_mode,
    )
    y_class = np.array([workshop_label_to_class(int(l)) for l in y_workshop], dtype=np.int64)
    return X, y_class, groups


def loso_test_subjects(groups: np.ndarray) -> list[int]:
    return sorted(np.unique(groups).tolist())


def validate_no_subject_overlap(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    train_s = set(np.unique(train_groups))
    test_s = set(np.unique(test_groups))
    overlap = train_s & test_s
    if overlap:
        raise ValueError(f"Subject leakage detected: {overlap}")
