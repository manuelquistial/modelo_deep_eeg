from physionet_mi.constants import (
    CLASS_TO_NAME,
    LABEL_NAME_TO_ID,
    LABEL_TO_CLASS,
    class_to_workshop_label,
    workshop_label_to_class,
)


def test_label_maps():
    assert LABEL_TO_CLASS["left_hand"] == 0
    assert LABEL_TO_CLASS["right_hand"] == 1
    assert workshop_label_to_class(LABEL_NAME_TO_ID["left_hand"]) == 0
    assert workshop_label_to_class(LABEL_NAME_TO_ID["right_hand"]) == 1
    assert class_to_workshop_label(0) == 1
    assert class_to_workshop_label(1) == 2
    assert CLASS_TO_NAME[0] == "left_hand"
    assert CLASS_TO_NAME[1] == "right_hand"
