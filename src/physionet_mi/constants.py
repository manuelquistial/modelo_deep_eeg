"""Label and dataset constants (aligned with workshop.ipynb)."""

BINARY_EVENTS = ["left_hand", "right_hand"]

# Workshop internal IDs (1=left, 2=right)
LABEL_NAME_TO_ID = {"left_hand": 1, "right_hand": 2}
LABEL_ORDER_WORKSHOP = [1, 2]
LABEL_NAMES = ["left_hand", "right_hand"]
LABEL_ID_TO_NAME = dict(zip(LABEL_ORDER_WORKSHOP, LABEL_NAMES))

# PyTorch class indices (0=left, 1=right)
LABEL_TO_CLASS = {"left_hand": 0, "right_hand": 1}
CLASS_TO_NAME = {0: "left_hand", 1: "right_hand"}
CLASS_ORDER = [0, 1]

SFREQ = 160

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


def workshop_label_to_class(label: int) -> int:
    """Map workshop label (1|2) to PyTorch class (0|1)."""
    if label == LABEL_NAME_TO_ID["left_hand"]:
        return 0
    if label == LABEL_NAME_TO_ID["right_hand"]:
        return 1
    raise ValueError(f"Unknown workshop label: {label}")


def class_to_workshop_label(class_idx: int) -> int:
    """Map PyTorch class (0|1) to workshop label (1|2)."""
    if class_idx == 0:
        return LABEL_NAME_TO_ID["left_hand"]
    if class_idx == 1:
        return LABEL_NAME_TO_ID["right_hand"]
    raise ValueError(f"Unknown class index: {class_idx}")
