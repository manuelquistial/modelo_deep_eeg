import numpy as np

from physionet_mi.evaluation.subject_splits import (
    assert_no_subject_overlap,
    get_subject_disjoint_holdout_split,
    make_groupkfold_splits,
    summarize_split,
)


def test_holdout_split_sizes():
    subjects = np.arange(20)
    labels = {i: i % 2 for i in subjects}
    dev, test = get_subject_disjoint_holdout_split(subjects, labels, 0.2, 42)
    assert len(set(dev) & set(test)) == 0
    assert len(dev) + len(test) == 20


def test_groupkfold_covers_all():
    subjects = np.arange(9)
    folds = make_groupkfold_splits(subjects, 3)
    assert len(folds) == 3
    seen_test = set()
    for test_ids, dev_ids in folds:
        assert_no_subject_overlap(dev_ids, np.array([]), test_ids)
        seen_test.update(test_ids.tolist())
    assert seen_test == set(subjects.tolist())


def test_summarize_split():
    groups = np.array([1, 1, 2, 2, 3, 3])
    y = np.array([0, 0, 1, 1, 0, 1])
    meta = summarize_split(
        dataset="test",
        seed=0,
        train_subjects=np.array([1, 2]),
        val_subjects=None,
        test_subjects=np.array([3]),
        y=y,
        groups=groups,
    )
    assert meta["train"]["n_subjects"] == 2
    assert meta["test"]["n_subjects"] == 1
