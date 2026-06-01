import numpy as np

from physionet_mi.config import ExperimentConfig, PreprocessConfig
from physionet_mi.data.preprocessing import crop_or_pad_trial_time, preprocess_train_eval_subject_dicts
from physionet_mi.data.subject_dict import split_subjects_holdout, to_model_arrays


def test_align_n_times_for_cnn():
    from physionet_mi.data.preprocessing import align_n_times_for_cnn

    assert align_n_times_for_cnn(481) == 480
    assert align_n_times_for_cnn(480) == 480


def test_crop_center():
    trial = np.ones((481, 64), dtype=np.float32)
    out = crop_or_pad_trial_time(trial, 480, mode="crop")
    assert out.shape == (480, 64)


def test_to_model_arrays(synthetic_subject_dict):
    X, y, groups = to_model_arrays(synthetic_subject_dict, 480, 64)
    assert X.shape == (8, 64, 480)
    assert set(y.tolist()) <= {0, 1}
    assert len(groups) == 8


def test_preprocess_pipeline(synthetic_subject_dict):
    dev_ids, test_ids = split_subjects_holdout(synthetic_subject_dict, 0.5, 42)
    dev = {int(s): synthetic_subject_dict[int(s)] for s in dev_ids}
    test = {int(s): synthetic_subject_dict[int(s)] for s in test_ids}
    cfg = ExperimentConfig()
    cfg.preprocess.use_ea = False
    payload = preprocess_train_eval_subject_dicts(dev, test, cfg, n_channels=64, verbose=False)
    X, y, g = to_model_arrays(payload["train_dict"], payload["common_n_times"], 64)
    assert X.ndim == 3
    assert X.shape[1] == 64
