import numpy as np
import pandas as pd

from physionet_mi.analysis.erd_ers import compute_trial_bandpower


def test_lateralization_columns():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 6, 128)).astype(np.float32)
    y = np.array([0, 0, 1, 1])
    groups = np.array([1, 1, 2, 2])
    ch_names = ["Fz", "FC3", "FCz", "FC4", "C3", "C4"]
    df = compute_trial_bandpower(X, y, groups, ch_names, sfreq=160.0)
    assert "lateralization_index_mu" in df.columns
    assert len(df) == 4
