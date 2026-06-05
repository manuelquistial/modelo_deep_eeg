"""Bootstrap confidence intervals for repeated evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI for 1D metric values."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    rng = np.random.default_rng(random_state)
    n = len(values)
    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = float(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(boots, [alpha, 1.0 - alpha])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(n),
    }


def aggregate_repeated_results(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate repeated runs with mean/std/median/min/max and bootstrap CI."""
    if metric_cols is None:
        metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "kappa"]
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_success"] = int((grp["status"] == "ok").sum()) if "status" in grp else len(grp)
        row["n_failed"] = int((grp["status"] != "ok").sum()) if "status" in grp else 0
        for col in metric_cols:
            if col not in grp.columns:
                continue
            ok = grp[grp["status"] == "ok"] if "status" in grp else grp
            stats = bootstrap_ci(ok[col].to_numpy())
            for k, v in stats.items():
                row[f"{col}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
