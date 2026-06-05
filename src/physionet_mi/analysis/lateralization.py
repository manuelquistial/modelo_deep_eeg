"""Correlate lateralization with subject-level accuracy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_subject_lateralization(trial_df: pd.DataFrame) -> pd.DataFrame:
    return (
        trial_df.groupby(["subject_id", "label"])
        .agg(
            mean_li_mu=("lateralization_index_mu", "mean"),
            mean_contra_mu=("contralateral_mu", "mean"),
            n_trials=("lateralization_index_mu", "count"),
        )
        .reset_index()
    )


def correlate_lateralization_accuracy(
    trial_df: pd.DataFrame,
    subject_acc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Pearson/Spearman between subject mean LI and subject accuracy."""
    subj_li = trial_df.groupby("subject_id")["lateralization_index_mu"].mean()
    merged = subject_acc_df.merge(
        subj_li.reset_index(),
        on="subject_id",
        how="inner",
    )
    rows = []
    for model, grp in merged.groupby("model"):
        x = grp["lateralization_index_mu"].to_numpy()
        y = grp["accuracy"].to_numpy()
        if len(grp) < 5:
            continue
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        rows.append({
            "model": model,
            "n_subjects": len(grp),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
        })
    return pd.DataFrame(rows)


def compare_correct_vs_error_lateralization(
    trial_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare LI for correct vs incorrect trials when predictions available."""
    if "subject_id" not in predictions_df.columns:
        return pd.DataFrame()
    merged = trial_df.reset_index(drop=True).copy()
    merged["trial_idx"] = np.arange(len(merged))
    pred = predictions_df.reset_index(drop=True)
    if len(pred) != len(merged):
        return pd.DataFrame()
    merged["correct"] = pred["y_true"] == pred["y_pred"]
    rows = []
    for label in merged["label"].unique():
        sub = merged[merged["label"] == label]
        corr = sub[sub["correct"]]["lateralization_index_mu"].dropna()
        err = sub[~sub["correct"]]["lateralization_index_mu"].dropna()
        if len(corr) < 3 or len(err) < 3:
            continue
        stat, p = stats.mannwhitneyu(corr, err, alternative="two-sided")
        rows.append({
            "label": label,
            "mean_li_correct": float(corr.mean()),
            "mean_li_error": float(err.mean()),
            "n_correct": len(corr),
            "n_error": len(err),
            "mannwhitney_p": float(p),
        })
    return pd.DataFrame(rows)
