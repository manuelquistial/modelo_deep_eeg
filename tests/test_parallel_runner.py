import os

import numpy as np

from physionet_mi.config import ExperimentConfig, ModelConfig
from physionet_mi.evaluation.parallel_runner import (
    _sync_model_dims_from_arrays,
    effective_inner_n_jobs,
    partition_models,
)
from physionet_mi.models.eegnet import EEGNet


def test_partition_models():
    classical, deep = partition_models(
        ["fbcsp_lda", "csp_svm", "eegnet", "riemann_mdm", "eegme"]
    )
    assert classical == ["fbcsp_lda", "csp_svm", "riemann_mdm"]
    assert deep == ["eegnet", "eegme"]


def test_effective_inner_n_jobs_serial():
    assert effective_inner_n_jobs(1) == -1


def test_effective_inner_n_jobs_parallel():
    n = effective_inner_n_jobs(4)
    assert n >= 1
    assert n <= (os.cpu_count() or 1)


def test_sync_model_dims_from_arrays_updates_eegnet_fc():
    cfg = ExperimentConfig()
    cfg.model = ModelConfig(n_channels=64, n_times=480)
    arrays = {
        "X_dev": np.zeros((4, 22, 384), dtype=np.float32),
        "meta": {"n_channels": 22, "model_n_times": 384},
    }
    _sync_model_dims_from_arrays(cfg, arrays)
    assert cfg.model.n_channels == 22
    assert cfg.model.n_times == 384

    model_synced = EEGNet(n_channels=22, n_times=384)
    x = np.zeros((2, 22, 384), dtype=np.float32)
    import torch

    batch = torch.from_numpy(x).unsqueeze(1)
    feat = model_synced._forward_features(batch)
    assert feat.shape[1] == model_synced.fc.in_features
    logits = model_synced.fc(feat)
    assert logits.shape == (2, 2)
