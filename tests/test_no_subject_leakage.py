import numpy as np

from physionet_mi.evaluation.subject_splits import assert_no_subject_overlap


def test_assert_overlap_raises():
    try:
        assert_no_subject_overlap(np.array([1, 2]), np.array([2, 3]), np.array([4]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_assert_no_overlap_passes():
    assert_no_subject_overlap(np.array([1, 2]), np.array([3]), np.array([4, 5]))
