"""Pytest fixtures with synthetic EEG trials."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_subject_dict():
    """Two subjects, 4 trials each, 64 channels, 480 samples."""
    rng = np.random.default_rng(42)
    subj_data = {}
    for sid in [1, 2]:
        trials = []
        labels = []
        for i in range(4):
            trials.append(rng.standard_normal((480, 64)).astype(np.float32))
            labels.append(1 if i % 2 == 0 else 2)
        subj_data[sid] = {
            "trials": trials,
            "labels": np.array(labels, dtype=int),
            "run_ids": np.arange(4, dtype=int),
            "ch_names": [f"Ch{i}" for i in range(64)],
        }
    return subj_data


@pytest.fixture
def synthetic_arrays():
    N, C, T = 16, 64, 480
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    y = np.array([0, 1] * (N // 2), dtype=np.int64)
    groups = np.array([1] * 8 + [2] * 8, dtype=np.int64)
    return X, y, groups
