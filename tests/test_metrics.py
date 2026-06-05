import numpy as np

from physionet_mi.evaluation.bootstrap import bootstrap_ci
from physionet_mi.evaluation.metrics import compute_metrics


def test_compute_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["kappa"] == 1.0


def test_bootstrap_ci():
    vals = np.array([0.6, 0.65, 0.7, 0.68, 0.72])
    stats = bootstrap_ci(vals, n_bootstrap=500, random_state=0)
    assert stats["n"] == 5
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
