"""Statistical tests for repeated hold-out and paired predictions."""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def _holm_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order, start=1):
        adjusted[idx] = min(1.0, p_values[idx] * (m - rank + 1))
    return adjusted.tolist()


def paired_wilcoxon_ea(
    df: pd.DataFrame,
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Paired Wilcoxon: EA vs no-EA per dataset/model across seeds."""
    rows = []
    for (dataset, model), grp in df.groupby(["dataset", "model"]):
        ea = grp[grp["use_ea"] == True].sort_values("seed")  # noqa: E712
        no = grp[grp["use_ea"] == False].sort_values("seed")  # noqa: E712
        merged = ea.merge(no, on="seed", suffixes=("_ea", "_no"))
        if len(merged) < 3:
            warnings.warn(f"Insufficient paired seeds for {dataset}/{model}")
            continue
        a = merged[f"{metric}_ea"].to_numpy()
        b = merged[f"{metric}_no"].to_numpy()
        try:
            stat, p = stats.wilcoxon(a, b, zero_method="wilcox")
        except ValueError as e:
            rows.append({
                "dataset": dataset, "model": model, "metric": metric,
                "n_pairs": len(merged), "p_value": np.nan, "error": str(e),
            })
            continue
        diff = a - b
        rows.append({
            "dataset": dataset,
            "model": model,
            "metric": metric,
            "n_pairs": len(merged),
            "mean_diff_ea_minus_no": float(np.mean(diff)),
            "median_diff": float(np.median(diff)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
        })
    return pd.DataFrame(rows)


def friedman_by_dataset_ea(
    df: pd.DataFrame,
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Friedman test comparing models within dataset×EA."""
    rows = []
    ok = df[df["status"] == "ok"] if "status" in df else df
    for (dataset, use_ea), grp in ok.groupby(["dataset", "use_ea"]):
        pivot = grp.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        pivot = pivot.dropna(axis=1, how="all").dropna(axis=0, how="any")
        if pivot.shape[1] < 3 or pivot.shape[0] < 3:
            warnings.warn(f"Skipping Friedman {dataset} EA={use_ea}: shape {pivot.shape}")
            continue
        try:
            stat, p = stats.friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
        except ValueError as e:
            rows.append({
                "dataset": dataset, "use_ea": use_ea, "metric": metric,
                "n_models": pivot.shape[1], "n_blocks": pivot.shape[0],
                "p_value": np.nan, "error": str(e),
            })
            continue
        rows.append({
            "dataset": dataset,
            "use_ea": bool(use_ea),
            "metric": metric,
            "n_models": int(pivot.shape[1]),
            "n_blocks": int(pivot.shape[0]),
            "friedman_stat": float(stat),
            "p_value": float(p),
            "models": list(pivot.columns),
        })
    return pd.DataFrame(rows)


def pairwise_wilcoxon_models(
    df: pd.DataFrame,
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Post-hoc paired Wilcoxon between models with Holm correction per group."""
    rows = []
    ok = df[df["status"] == "ok"] if "status" in df else df
    for (dataset, use_ea), grp in ok.groupby(["dataset", "use_ea"]):
        pivot = grp.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        models = [c for c in pivot.columns if pivot[c].notna().sum() >= 3]
        pairs: list[tuple[str, str, float, float]] = []
        for i, m1 in enumerate(models):
            for m2 in models[i + 1 :]:
                sub = pivot[[m1, m2]].dropna()
                if len(sub) < 3:
                    continue
                try:
                    stat, p = stats.wilcoxon(sub[m1], sub[m2], zero_method="wilcox")
                    pairs.append((m1, m2, float(stat), float(p)))
                except ValueError:
                    continue
        if not pairs:
            continue
        pvals = [p for *_, p in pairs]
        adj = _holm_correction(pvals)
        for (m1, m2, stat, p), p_adj in zip(pairs, adj):
            rows.append({
                "dataset": dataset,
                "use_ea": bool(use_ea),
                "metric": metric,
                "model_a": m1,
                "model_b": m2,
                "wilcoxon_stat": stat,
                "p_value": p,
                "p_value_holm": p_adj,
            })
    return pd.DataFrame(rows)


def mcnemar_from_predictions(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """McNemar test for two classifiers on same trials."""
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    result = stats.binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided")
    return {"b": b, "c": c, "p_value": float(result.pvalue)}


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) < 2:
        return np.nan
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
