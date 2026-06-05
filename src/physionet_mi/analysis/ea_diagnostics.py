"""Euclidean Alignment diagnostics — covariance dispersion before/after EA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from physionet_mi.data.preprocessing import apply_euclidean_alignment_array, mean_covariance_ea, trial_covariance_ea


def _frobenius_dist(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, "fro"))


def covariance_distance_stats(X: np.ndarray, groups: np.ndarray, reg: float = 1e-10) -> dict:
    """Within- and between-subject mean Frobenius distance of trial covariances."""
    covs = [_trial_cov(X[i], reg) for i in range(len(X))]
    subjects = np.unique(groups)
    within = []
    for sid in subjects:
        idx = np.where(groups == sid)[0]
        if len(idx) < 2:
            continue
        sub_covs = [covs[i] for i in idx]
        ref = np.mean(sub_covs, axis=0)
        within.extend([_frobenius_dist(c, ref) for c in sub_covs])
    between = []
    sub_means = []
    for sid in subjects:
        idx = np.where(groups == sid)[0]
        sub_means.append(np.mean([covs[i] for i in idx], axis=0))
    global_mean = np.mean(sub_means, axis=0)
    between = [_frobenius_dist(m, global_mean) for m in sub_means]
    return {
        "within_subject_mean": float(np.mean(within)) if within else np.nan,
        "within_subject_std": float(np.std(within)) if within else np.nan,
        "between_subject_mean": float(np.mean(between)) if between else np.nan,
        "between_subject_std": float(np.std(between)) if between else np.nan,
    }


def _trial_cov(trial_ct: np.ndarray, reg: float) -> np.ndarray:
    return trial_covariance_ea(trial_ct, reg=reg)


def compare_ea_dispersion(
    X: np.ndarray,
    groups: np.ndarray,
    reg: float = 1e-10,
) -> tuple[dict, dict, pd.DataFrame]:
    """Align with EA reference from all trials; return before/after stats."""
    before = covariance_distance_stats(X, groups, reg=reg)
    R = mean_covariance_ea(X, reg=reg)
    from physionet_mi.data.preprocessing import invsqrt_spd

    W = invsqrt_spd(R)
    X_aligned = apply_euclidean_alignment_array(X, W)
    after = covariance_distance_stats(X_aligned, groups, reg=reg)
    row = {
        "within_before": before["within_subject_mean"],
        "within_after": after["within_subject_mean"],
        "between_before": before["between_subject_mean"],
        "between_after": after["between_subject_mean"],
        "within_reduction_pct": (
            100 * (before["within_subject_mean"] - after["within_subject_mean"]) / (before["within_subject_mean"] + 1e-12)
        ),
        "between_reduction_pct": (
            100 * (before["between_subject_mean"] - after["between_subject_mean"]) / (before["between_subject_mean"] + 1e-12)
        ),
    }
    return before, after, pd.DataFrame([row])
