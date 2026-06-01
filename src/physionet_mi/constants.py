"""Dataset-specific constants and helpers."""

from __future__ import annotations

import numpy as np

BINARY_EVENTS = ["left_hand", "right_hand"]

LABEL_NAME_TO_ID = {"left_hand": 1, "right_hand": 2}
LABEL_ORDER_WORKSHOP = [1, 2]
LABEL_NAMES = ["left_hand", "right_hand"]
LABEL_ID_TO_NAME = dict(zip(LABEL_ORDER_WORKSHOP, LABEL_NAMES))

LABEL_TO_CLASS = {"left_hand": 0, "right_hand": 1}
CLASS_TO_NAME = {0: "left_hand", 1: "right_hand"}
CLASS_ORDER = [0, 1]

SFREQ_PHYSIONET = 160
SFREQ_BNCI2014_001 = 125

# Backward compatibility
SFREQ = SFREQ_PHYSIONET

DEFAULT_FREQ_BANDS = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
    (32, 36),
]

SUPPORTED_DATASETS = ("physionet", "bnci2014_001")


def workshop_label_to_class(label: int) -> int:
    if label == LABEL_NAME_TO_ID["left_hand"]:
        return 0
    if label == LABEL_NAME_TO_ID["right_hand"]:
        return 1
    raise ValueError(f"Unknown workshop label: {label}")


def class_to_workshop_label(class_idx: int) -> int:
    if class_idx == 0:
        return LABEL_NAME_TO_ID["left_hand"]
    if class_idx == 1:
        return LABEL_NAME_TO_ID["right_hand"]
    raise ValueError(f"Unknown class index: {class_idx}")


def class_indices_to_workshop(y_class: np.ndarray) -> np.ndarray:
    return np.array([class_to_workshop_label(int(c)) for c in y_class], dtype=np.int64)
