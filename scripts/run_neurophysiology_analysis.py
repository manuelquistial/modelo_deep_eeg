#!/usr/bin/env python3
"""Mu/beta lateralization and band-power analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.analysis.erd_ers import compute_trial_bandpower
from physionet_mi.analysis.lateralization import (
    aggregate_subject_lateralization,
    compare_correct_vs_error_lateralization,
    correlate_lateralization_accuracy,
)
from physionet_mi.analysis.subject_level import collect_subject_level_metrics
from physionet_mi.config import load_config, dataset_sfreq
from physionet_mi.data.cache import load_holdout_arrays
from physionet_mi.evaluation.config_matrix import resolve_config_path
from physionet_mi.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["physionet", "bnci2014_001", "bnci"])
    p.add_argument("--ea", default="both")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = "bnci2014_001" if args.dataset == "bnci" else args.dataset
    ea_modes = [True, False] if args.ea == "both" else [args.ea.lower() in ("true", "1", "ea")]

    all_trials = []
    for use_ea in ea_modes:
        preprocess = "ea" if use_ea else "no_ea"
        cfg = load_config(resolve_config_path(dataset, preprocess, ROOT), project_root=ROOT)
        data = load_holdout_arrays(cfg)
        ch_names = data["meta"].get("ch_names", [])
        sfreq = dataset_sfreq(cfg)
        X = __import__("numpy").concatenate([data["X_dev"], data["X_test"]])
        y = __import__("numpy").concatenate([data["y_dev"], data["y_test"]])
        g = __import__("numpy").concatenate([data["groups_dev"], data["groups_test"]])
        trial_df = compute_trial_bandpower(X, y, g, ch_names, sfreq)
        trial_df["use_ea"] = use_ea
        trial_df["dataset"] = dataset
        all_trials.append(trial_df)

    trial_all = pd.concat(all_trials, ignore_index=True)
    trial_all.to_csv(args.output_dir / "erd_ers_trial_level.csv", index=False)
    subj = aggregate_subject_lateralization(trial_all)
    subj.to_csv(args.output_dir / "erd_ers_subject_level.csv", index=False)

    from physionet_mi.paths import publishable_runs_dir

    pred_root = publishable_runs_dir(ROOT) / "repeated_holdout" / dataset.replace("bnci2014_001", "bnci")
    if pred_root.exists():
        subj_acc = collect_subject_level_metrics(pred_root)
        corr = correlate_lateralization_accuracy(trial_all, subj_acc)
        corr.to_csv(args.output_dir / "lateralization_vs_accuracy.csv", index=False)

    summary = [
        "# Neurophysiology summary\n",
        "Relative mu/beta band power and lateralization index (not baseline-corrected ERD%).\n",
        f"- Trials analyzed: {len(trial_all)}\n",
    ]
    (args.output_dir / "neurophysiology_summary.md").write_text("".join(summary), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, grp in trial_all.groupby("label"):
        ax.hist(grp["lateralization_index_mu"].dropna(), bins=30, alpha=0.5, label=label)
    ax.legend()
    ax.set_xlabel("Mu lateralization index")
    fig.savefig(args.output_dir / "lateralization_distribution_by_class.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
