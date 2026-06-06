import json
import tempfile
from pathlib import Path

import numpy as np

from physionet_mi.evaluation.bootstrap import bootstrap_ci
from physionet_mi.evaluation.metrics import compute_metrics
from physionet_mi.evaluation.reporting import save_run_artifacts


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


def test_save_run_artifacts_metrics_json_serializable():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        save_run_artifacts(
            out,
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 1, 0]),
            {"model": "test"},
            extra={"groups": np.array([10, 10, 20, 20]), "model": "fbcsp_lda"},
        )
        data = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert "groups" not in data
        assert data["accuracy"] == 0.5
