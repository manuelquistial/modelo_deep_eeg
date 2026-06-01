"""BNCI loader label encoding tests."""

from physionet_mi.data.bnci_loader import _encode_labels, _normalize_event_label


def test_normalize_event_label_string():
    assert _normalize_event_label("left_hand") == "left_hand"
    assert _normalize_event_label(1) == "left_hand"


def test_encode_labels_filters_feet():
    import numpy as np

    y = np.array(["left_hand", "right_hand", "feet"])
    encoded, keep = _encode_labels(y, ["left_hand", "right_hand"])
    assert keep.sum() == 2
    assert list(encoded) == [0, 1]
