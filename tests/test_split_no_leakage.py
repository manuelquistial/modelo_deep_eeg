import numpy as np

from physionet_mi.data.subject_dict import (
    mask_by_subjects,
    split_subjects_holdout,
    split_val_subjects_stratified,
    validate_no_subject_overlap,
)


def test_holdout_no_overlap(synthetic_subject_dict):
    dev_ids, test_ids = split_subjects_holdout(synthetic_subject_dict, 0.5, 42)
    assert len(set(dev_ids) & set(test_ids)) == 0


def test_val_split_no_leakage():
    # Each subject has a single majority class (realistic for MI cohorts).
    groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8])
    y_w = np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
    train_subj, val_subj = split_val_subjects_stratified(groups, y_w, 0.15, 42)
    assert len(val_subj) >= 2
    tr = groups[mask_by_subjects(groups, train_subj)]
    va = groups[mask_by_subjects(groups, val_subj)]
    validate_no_subject_overlap(tr, va)
