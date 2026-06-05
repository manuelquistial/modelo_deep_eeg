#!/usr/bin/env python3
"""Euclidean Alignment covariance diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.analysis.ea_diagnostics import compare_ea_dispersion
from physionet_mi.config import load_config
from physionet_mi.data.cache import load_holdout_arrays
from physionet_mi.evaluation.config_matrix import resolve_config_path
from physionet_mi.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["physionet", "bnci2014_001", "bnci"])
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = "bnci2014_001" if args.dataset == "bnci" else args.dataset
    cfg = load_config(resolve_config_path(dataset, "no_ea", ROOT), project_root=ROOT)
    data = load_holdout_arrays(cfg)
    import numpy as np
    X = np.concatenate([data["X_dev"], data["X_test"]])
    g = np.concatenate([data["groups_dev"], data["groups_test"]])

    _, _, df = compare_ea_dispersion(X, g, reg=cfg.preprocess.ea_reg)
    df["dataset"] = dataset
    df.to_csv(args.output_dir / "ea_covariance_distances.csv", index=False)

    fixed = ROOT / "outputs" / "pipeline_comparison.csv"
    if fixed.exists():
        comp = pd.read_csv(fixed)
        comp.groupby(["dataset", "model"]).apply(
            lambda x: x.loc[x["use_ea"]].accuracy.mean() - x.loc[~x["use_ea"]].accuracy.mean()
            if "use_ea" in x.columns else 0
        )
        gain_rows = []
        for (ds, model), grp in comp.groupby(["dataset", "model"]):
            ea = grp[grp["use_ea"] == True]  # noqa
            no = grp[grp["use_ea"] == False]  # noqa
            if len(ea) and len(no):
                gain_rows.append({
                    "dataset": ds, "model": model,
                    "ea_gain_accuracy": float(ea["accuracy"].iloc[0] - no["accuracy"].iloc[0]),
                })
        if gain_rows:
            pd.DataFrame(gain_rows).to_csv(args.output_dir / "ea_gain_per_model.csv", index=False)

    (args.output_dir / "ea_diagnostics_summary.md").write_text(
        "# EA diagnostics\n\nCovariance dispersion before/after EA alignment.\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
