import pytest

torch = pytest.importorskip("torch")

from physionet_mi.models.eegnet import EEGNet


def test_eegnet_forward_shape():
    model = EEGNet(n_channels=64, n_times=480)
    x = torch.randn(4, 1, 64, 480)
    logits, emb = model(x)
    assert logits.shape == (4, 2)
    assert emb.shape[0] == 4


def test_eegnet_registry_build():
    from physionet_mi.config import ExperimentConfig
    from physionet_mi.models.registry import build_model

    cfg = ExperimentConfig()
    cfg.model.name = "eegnet"
    cfg.model.n_channels = 64
    cfg.model.n_times = 480
    model = build_model(cfg)
    assert isinstance(model, EEGNet)
