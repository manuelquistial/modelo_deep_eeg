"""Subject-wise GroupKFold evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from physionet_mi.config import load_config
from physionet_mi.data.cache import build_arrays_for_subject_split, load_raw_subject_dict
from physionet_mi.evaluation.config_matrix import resolve_config_path
from physionet_mi.evaluation.runners import ALL_MODELS, evaluate_model_on_arrays
from physionet_mi.evaluation.subject_splits import make_groupkfold_splits, save_split_metadata, summarize_split

logger = logging.getLogger(__name__)


def _should_skip_existing_run(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        import json

        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if data.get("status") == "error":
        return False
    return "accuracy" in data


def run_groupkfold(
    *,
    dataset: str,
    n_splits: int,
    models: Iterable[str],
    ea_modes: Iterable[bool],
    output_dir: Path,
    project_root: Path,
    skip_deep: bool = False,
    skip_existing: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for use_ea in ea_modes:
        preprocess = "ea" if use_ea else "no_ea"
        cfg_path = resolve_config_path(dataset, preprocess, project_root)
        cfg = load_config(cfg_path, project_root=project_root)
        subj_data, ch_names, _ = load_raw_subject_dict(cfg)
        all_subjects = np.array(sorted(subj_data.keys()), dtype=int)
        folds = make_groupkfold_splits(all_subjects, n_splits)

        for fold_idx, (test_ids, dev_ids) in enumerate(folds):
            arrays = build_arrays_for_subject_split(cfg, subj_data, dev_ids, test_ids, ch_names)

            for model_name in models:
                if model_name not in ALL_MODELS:
                    continue
                if skip_deep and model_name in {"eegnet", "eegme"}:
                    continue

                run_tag = f"{model_name}_{preprocess}_fold{fold_idx}"
                run_dir = output_dir / run_tag
                if skip_existing and _should_skip_existing_run(run_dir):
                    continue

                split_meta = summarize_split(
                    dataset=dataset,
                    fold=fold_idx,
                    train_subjects=dev_ids,
                    val_subjects=np.array([], dtype=int),
                    test_subjects=test_ids,
                    y=np.concatenate([arrays["y_dev"], arrays["y_test"]]),
                    groups=np.concatenate([arrays["groups_dev"], arrays["groups_test"]]),
                )
                save_split_metadata(run_dir / "fold_metadata.json", split_meta)

                metrics = evaluate_model_on_arrays(
                    model_name, cfg, arrays, run_dir, protocol="groupkfold"
                )
                rows.append({
                    "dataset": dataset,
                    "fold": fold_idx,
                    "model": model_name,
                    "use_ea": use_ea,
                    "n_dev_subjects": len(dev_ids),
                    "n_test_subjects": len(test_ids),
                    "n_test_trials": len(arrays["y_test"]),
                    "accuracy": metrics.get("accuracy"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "kappa": metrics.get("kappa"),
                    "status": metrics.get("status", "ok"),
                    "error_message": metrics.get("error_message", ""),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "groupkfold_results.csv", index=False)
    if len(df):
        summary = df.groupby(["dataset", "model", "use_ea"]).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            bal_acc_mean=("balanced_accuracy", "mean"),
            bal_acc_std=("balanced_accuracy", "std"),
            n_folds=("fold", "count"),
        ).reset_index()
        summary.to_csv(output_dir / "groupkfold_summary.csv", index=False)
    return df
