"""Preprocessing ported from workshop.ipynb."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, sosfiltfilt

from physionet_mi.config import ExperimentConfig, PreprocessConfig
from physionet_mi.constants import SFREQ
from physionet_mi.data.moabb_loader import SubjectRecord
from physionet_mi.data.subject_dict import (
    flatten_subject_dict,
    subset_subject_dict,
    summarize_subject_dict,
    trials_list_to_3d,
)

logger = logging.getLogger(__name__)

# EEGMeModel uses AvgPool2d(kernel=(1, 10)); temporal length must be divisible by 10.
CNN_POOL_FACTOR = 10


def align_n_times_for_cnn(n_times: int, pool_factor: int = CNN_POOL_FACTOR) -> int:
    """Crop length to the largest multiple of pool_factor (center crop applied downstream)."""
    aligned = (int(n_times) // pool_factor) * pool_factor
    if aligned < pool_factor:
        raise ValueError(f"Cannot align n_times={n_times} to pool_factor={pool_factor}")
    return aligned


def crop_or_pad_trial_time(trial: np.ndarray, target_n_times: int, mode: str = "crop") -> np.ndarray:
    n_times, n_channels = trial.shape
    if n_times == target_n_times:
        return trial.astype(np.float32, copy=False)
    if n_times > target_n_times:
        start = (n_times - target_n_times) // 2
        return trial[start : start + target_n_times, :].astype(np.float32, copy=False)
    if mode == "pad":
        out = np.zeros((target_n_times, n_channels), dtype=np.float32)
        out[:n_times, :] = trial.astype(np.float32, copy=False)
        return out
    raise ValueError(
        f"Trial n_times={n_times} < target={target_n_times}. Use mode='pad' or check epochs."
    )


def harmonize_subject_dict_time(
    data_dict: dict[int, SubjectRecord],
    target_n_times: int | None = None,
    crop_mode: str = "crop",
    verbose: bool = False,
    name: str = "data",
) -> tuple[dict[int, SubjectRecord], int]:
    all_trials = [trial for sdata in data_dict.values() for trial in sdata["trials"]]
    n_times_all = [trial.shape[0] for trial in all_trials]
    if target_n_times is None:
        target_n_times = int(np.min(n_times_all))

    if verbose:
        logger.info("[%s] harmonizing to %d samples", name, target_n_times)

    out: dict[int, SubjectRecord] = {}
    for sid, sdata in data_dict.items():
        fixed_trials = [
            crop_or_pad_trial_time(trial, target_n_times, mode=crop_mode) for trial in sdata["trials"]
        ]
        out[sid] = _copy_subject_shell(sdata, trials=fixed_trials)
    return out, target_n_times


def infer_signal_unit_for_outliers(data_dict: dict[int, SubjectRecord], max_trials: int = 1000) -> tuple[bool, np.ndarray]:
    values = []
    for sdata in data_dict.values():
        for trial in sdata["trials"]:
            values.append(float(np.max(np.abs(trial))))
            if len(values) >= max_trials:
                break
        if len(values) >= max_trials:
            break

    values = np.asarray(values)
    p = np.percentile(values, [50, 90, 95, 99, 100])
    multiply_by_1e6 = bool(p[2] < 1.0)
    return multiply_by_1e6, p


def _copy_subject_shell(
    sdata: SubjectRecord,
    trials: list[np.ndarray] | None = None,
    labels: np.ndarray | None = None,
    run_ids: np.ndarray | None = None,
    extra: dict | None = None,
) -> SubjectRecord:
    new = {k: v for k, v in sdata.items() if k not in ["trials", "labels", "run_ids"]}
    new["trials"] = trials if trials is not None else list(sdata["trials"])
    new["labels"] = np.asarray(labels if labels is not None else sdata["labels"], dtype=int)
    new["run_ids"] = np.asarray(run_ids if run_ids is not None else sdata["run_ids"], dtype=int)
    if extra:
        new.update(extra)
    return new


def remove_outliers_auto(
    data_dict: dict[int, SubjectRecord],
    threshold_uv: float = 800.0,
    auto_detect: bool = True,
    verbose: bool = True,
) -> tuple[dict[int, SubjectRecord], dict, dict]:
    multiply_by_1e6 = False
    percentiles = None
    if auto_detect:
        multiply_by_1e6, percentiles = infer_signal_unit_for_outliers(data_dict)

    cleaned: dict[int, SubjectRecord] = {}
    summary: dict = {}

    for sid, sdata in data_dict.items():
        keep_trials, keep_labels, keep_runs = [], [], []
        removed = 0
        for i, trial in enumerate(sdata["trials"]):
            max_abs_uv = np.max(np.abs(trial)) * (1e6 if multiply_by_1e6 else 1.0)
            if max_abs_uv <= threshold_uv:
                keep_trials.append(trial.astype(np.float32, copy=False))
                keep_labels.append(sdata["labels"][i])
                keep_runs.append(sdata["run_ids"][i])
            else:
                removed += 1
        cleaned[sid] = _copy_subject_shell(sdata, keep_trials, keep_labels, keep_runs)
        summary[sid] = {
            "initial": len(sdata["trials"]),
            "remaining": len(keep_trials),
            "removed": removed,
        }
        if verbose:
            logger.info(
                "Subject %s: initial=%d removed=%d remaining=%d",
                sid,
                summary[sid]["initial"],
                removed,
                summary[sid]["remaining"],
            )

    return cleaned, summary, {
        "multiply_by_1e6": multiply_by_1e6,
        "percentiles": percentiles,
        "threshold_uv": threshold_uv,
    }


def apply_highpass_filter(
    data_dict: dict[int, SubjectRecord],
    sfreq: float,
    highpass_hz: float,
    filter_order: int,
    verbose: bool = False,
) -> dict[int, SubjectRecord]:
    filtered: dict[int, SubjectRecord] = {}
    sos = butter(filter_order, highpass_hz, btype="highpass", fs=sfreq, output="sos")
    for sid, sdata in data_dict.items():
        filtered_trials = []
        for trial in sdata["trials"]:
            filtered_trials.append(sosfiltfilt(sos, trial, axis=0).astype(np.float32))
        filtered[sid] = _copy_subject_shell(sdata, trials=filtered_trials)
        if verbose:
            logger.info("Subject %s: HPF on %d trials", sid, len(filtered_trials))
    return filtered


def _regularize_cov(C: np.ndarray, reg: float) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)
    return C + reg * np.eye(C.shape[0])


def trial_covariance_ea(trial_ch_time: np.ndarray, reg: float) -> np.ndarray:
    X = np.asarray(trial_ch_time, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    C = (X @ X.T) / max(X.shape[1], 1)
    return _regularize_cov(C, reg=reg)


def mean_covariance_ea(X: np.ndarray, reg: float) -> np.ndarray:
    covs = np.asarray([trial_covariance_ea(trial, reg=0.0) for trial in X])
    C_ref = np.mean(covs, axis=0)
    return _regularize_cov(C_ref, reg=reg)


def invsqrt_spd(C: np.ndarray) -> np.ndarray:
    W = fractional_matrix_power(C, -0.5).real
    return W.astype(np.float32)


def apply_euclidean_alignment_array(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.asarray([W @ trial for trial in X], dtype=np.float32)


def verify_ea_identity_from_array(X_aligned: np.ndarray) -> dict[str, float]:
    C_mean = mean_covariance_ea(X_aligned, reg=0.0)
    I = np.eye(C_mean.shape[0])
    diff = C_mean - I
    diag = np.diag(C_mean)
    off = C_mean - np.diag(diag)
    return {
        "diag_mean": float(np.mean(diag)),
        "offdiag_abs_mean": float(np.mean(np.abs(off))),
        "fro_error": float(np.linalg.norm(diff, ord="fro")),
    }


def apply_ea_to_subject_dict_per_subject(
    data_dict: dict[int, SubjectRecord],
    target_n_times: int,
    n_channels: int,
    ea_reg: float,
    crop_mode: str = "crop",
    verbose: bool = False,
    name: str = "EA",
) -> tuple[dict[int, SubjectRecord], dict, dict]:
    aligned_dict: dict[int, SubjectRecord] = {}
    transforms: dict = {}
    diagnostics: dict = {}

    for sid, sdata in data_dict.items():
        X = trials_list_to_3d(
            sdata["trials"],
            target_n_times=target_n_times,
            target_n_channels=n_channels,
            crop_mode=crop_mode,
        )
        C_ref = mean_covariance_ea(X, reg=ea_reg)
        W = invsqrt_spd(C_ref)
        X_aligned = apply_euclidean_alignment_array(X, W)
        trials_aligned = [X_aligned[i].T.astype(np.float32) for i in range(X_aligned.shape[0])]
        aligned_dict[sid] = _copy_subject_shell(sdata, trials=trials_aligned)
        transforms[sid] = W
        diagnostics[sid] = verify_ea_identity_from_array(X_aligned)
        if verbose:
            d = diagnostics[sid]
            logger.info(
                "[%s] subject %s diag_mean=%.4f fro_error=%.4e",
                name,
                sid,
                d["diag_mean"],
                d["fro_error"],
            )
    return aligned_dict, transforms, diagnostics


def preprocess_train_eval_subject_dicts(
    train_dict: dict[int, SubjectRecord],
    eval_dict: dict[int, SubjectRecord],
    cfg: ExperimentConfig,
    n_channels: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """Full preprocess pipeline for train and eval subject dicts."""
    pp = cfg.preprocess
    variant = "ea" if pp.use_ea else "no_ea"

    train_clean, _, _ = remove_outliers_auto(
        train_dict,
        threshold_uv=pp.outlier_uv,
        auto_detect=pp.auto_detect_outlier_units,
        verbose=verbose,
    )
    eval_clean, _, _ = remove_outliers_auto(
        eval_dict,
        threshold_uv=pp.outlier_uv,
        auto_detect=pp.auto_detect_outlier_units,
        verbose=False,
    )

    train_hp = apply_highpass_filter(
        train_clean, SFREQ, pp.highpass_hz, pp.highpass_order, verbose=verbose
    )
    eval_hp = apply_highpass_filter(
        eval_clean, SFREQ, pp.highpass_hz, pp.highpass_order, verbose=False
    )

    train_trials = [t for s in train_hp.values() for t in s["trials"]]
    eval_trials = [t for s in eval_hp.values() for t in s["trials"]]
    common_n_times = int(min(t.shape[0] for t in train_trials + eval_trials))
    logger.info("[%s] common_n_times=%d", variant, common_n_times)

    train_hp, _ = harmonize_subject_dict_time(
        train_hp, target_n_times=common_n_times, crop_mode=pp.crop_mode, verbose=verbose, name="DEV"
    )
    eval_hp, _ = harmonize_subject_dict_time(
        eval_hp, target_n_times=common_n_times, crop_mode=pp.crop_mode, verbose=verbose, name="TEST"
    )

    ea_diagnostics: dict = {"train": {}, "eval": {}}
    if pp.use_ea:
        train_proc, _, train_diag = apply_ea_to_subject_dict_per_subject(
            train_hp,
            common_n_times,
            n_channels,
            pp.ea_reg,
            crop_mode=pp.crop_mode,
            verbose=verbose,
            name="DEV EA",
        )
        eval_proc, _, eval_diag = apply_ea_to_subject_dict_per_subject(
            eval_hp,
            common_n_times,
            n_channels,
            pp.ea_reg,
            crop_mode=pp.crop_mode,
            verbose=verbose,
            name="TEST EA",
        )
        ea_diagnostics = {"train": train_diag, "eval": eval_diag}
        if pp.ea_verify_enabled and train_diag:
            d0 = next(iter(train_diag.values()))
            logger.info(
                "EA verify DEV diag_mean=%.4f fro_error=%.4e",
                d0["diag_mean"],
                d0["fro_error"],
            )
    else:
        train_proc = train_hp
        eval_proc = eval_hp

    summarize_subject_dict("DEV processed", train_proc)
    summarize_subject_dict("TEST processed", eval_proc)

    return {
        "train_dict": train_proc,
        "eval_dict": eval_proc,
        "common_n_times": common_n_times,
        "ea_diagnostics": ea_diagnostics,
    }
