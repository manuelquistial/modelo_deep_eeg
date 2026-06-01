import pytest

torch = pytest.importorskip("torch")

from physionet_mi.models.eegme import EEGMeModel


def test_forward_shape():
    model = EEGMeModel(C=64, T=480, f=7, e=32, num_classes=2)
    x = torch.randn(4, 1, 64, 480)
    logits, emb, local, global_f = model(x)
    assert logits.shape == (4, 2)
    assert emb.shape == (4, 32)


def test_invalid_t_raises():
    try:
        EEGMeModel(C=64, T=481)
        assert False, "expected ValueError"
    except ValueError:
        pass
