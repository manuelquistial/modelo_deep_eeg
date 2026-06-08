"""Shared constants and helpers for manuscript figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_LABELS: dict[str, str] = {
    "csp_svm": "CSP+SVM",
    "fbcsp_lda": "FBCSP+LDA",
    "riemann_mdm": "Riemann MDM",
    "riemann_ts_lr": "Riemann TS+LR",
    "eegnet": "EEGNet",
}

MODEL_ORDER: list[str] = [
    "csp_svm",
    "fbcsp_lda",
    "riemann_ts_lr",
    "riemann_mdm",
    "eegnet",
]

DATASET_TITLES: dict[str, str] = {
    "physionet": "PhysioNet MI",
    "bnci": "BNCI2014-001",
    "bnci2014_001": "BNCI2014-001",
}


def dataset_key(value: str) -> str:
    v = str(value).lower()
    if "bnci" in v:
        return "bnci"
    if "physionet" in v:
        return "physionet"
    return v


def panel_title(value: str) -> str:
    return DATASET_TITLES.get(dataset_key(value), str(value))


def save_figure(fig, stem: Path) -> tuple[Path, Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def load_holdout_results(pub_root: Path, ds: str) -> pd.DataFrame | None:
    path = pub_root / "repeated_holdout" / ds / "repeated_holdout_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["status"] == "ok"] if "status" in df.columns else df


def rank_dataframe(df: pd.DataFrame, value_col: str = "balanced_accuracy") -> pd.DataFrame:
    """Average ranks across repeat_id (1=best)."""
    rows = []
    for repeat_id, grp in df.groupby("repeat_id"):
        for use_ea in (False, True):
            sub = grp[grp["use_ea"] == use_ea].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(value_col, ascending=False)
            sub["rank"] = np.arange(1, len(sub) + 1)
            rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
