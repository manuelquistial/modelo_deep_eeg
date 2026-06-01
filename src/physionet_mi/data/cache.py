"""Cache preprocessed arrays to disk."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np

from physionet_mi.config import ExperimentConfig, dataset_sfreq
from physionet_mi.data.cohort_loader import load_cohort
from physionet_mi.data.preprocessing import align_n_times_for_cnn, preprocess_train_eval_subject_dicts
from physionet_mi.data.subject_dict import (
    split_subjects_holdout,
    subset_subject_dict,
    to_model_arrays,
)

logger = logging.getLogger(__name__)


def _cache_key(cfg: ExperimentConfig, n_subjects: int) -> str:
    subject_part = (
        "all"
        if cfg.data.subject_ids is None
        else hashlib.md5(str(sorted(cfg.data.subject_ids)).encode()).hexdigest()[:8]
    )
    ea = "ea" if cfg.preprocess.use_ea else "no_ea"
    ds = cfg.data.dataset.replace("/", "_")
    return f"{ds}_lr_{ea}_{n_subjects}sub_{subject_part}"


def cache_dir_for(cfg: ExperimentConfig, n_subjects: int) -> Path:
    return cfg.cache_path() / _cache_key(cfg, n_subjects)


def build_and_cache_holdout(cfg: ExperimentConfig, force: bool = False) -> Path:
    """Load MOABB, preprocess, split hold-out, save npz arrays."""
    out_dir = cache_dir_for(cfg, n_subjects=0)  # placeholder, updated after load

    subj_data, ch_names = load_cohort(cfg)
    n_channels = len(ch_names)
    out_dir = cache_dir_for(cfg, len(subj_data))
    holdout_dir = out_dir / "holdout"
    meta_path = out_dir / "meta.json"

    if holdout_dir.exists() and meta_path.exists() and not force:
        logger.info("Cache exists at %s", out_dir)
        return out_dir

    holdout_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_ids, test_ids = split_subjects_holdout(
        subj_data, cfg.split.test_size, cfg.split.random_state
    )
    dev_dict = subset_subject_dict(subj_data, dev_ids)
    test_dict = subset_subject_dict(subj_data, test_ids)

    payload = preprocess_train_eval_subject_dicts(
        dev_dict, test_dict, cfg, n_channels=n_channels, verbose=True
    )
    common_n_times_raw = payload["common_n_times"]
    model_n_times = align_n_times_for_cnn(common_n_times_raw)
    if model_n_times != common_n_times_raw:
        logger.info(
            "Adjusted n_times for CNN pooling: %d -> %d",
            common_n_times_raw,
            model_n_times,
        )
    cfg.model.n_times = model_n_times
    cfg.model.n_channels = n_channels

    X_dev, y_dev, groups_dev = to_model_arrays(
        payload["train_dict"], model_n_times, n_channels, cfg.preprocess.crop_mode
    )
    X_test, y_test, groups_test = to_model_arrays(
        payload["eval_dict"], model_n_times, n_channels, cfg.preprocess.crop_mode
    )

    np.save(holdout_dir / "X_dev.npy", X_dev)
    np.save(holdout_dir / "y_dev.npy", y_dev)
    np.save(holdout_dir / "groups_dev.npy", groups_dev)
    np.save(holdout_dir / "X_test.npy", X_test)
    np.save(holdout_dir / "y_test.npy", y_test)
    np.save(holdout_dir / "groups_test.npy", groups_test)
    np.save(holdout_dir / "dev_subject_ids.npy", dev_ids)
    np.save(holdout_dir / "test_subject_ids.npy", test_ids)

    joblib.dump(subj_data, out_dir / "subject_dict_raw.joblib")

    meta = {
        "dataset": cfg.data.dataset,
        "n_subjects": len(subj_data),
        "n_channels": n_channels,
        "sfreq": dataset_sfreq(cfg),
        "common_n_times": model_n_times,
        "common_n_times_raw": common_n_times_raw,
        "model_n_times": model_n_times,
        "use_ea": cfg.preprocess.use_ea,
        "dev_subjects": dev_ids.tolist(),
        "test_subjects": test_ids.tolist(),
        "ch_names": ch_names,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Cached hold-out data at %s", out_dir)
    return out_dir


def _pick_holdout_cache_dir(cfg: ExperimentConfig) -> Path | None:
    """Select the best matching cache directory for this config."""
    cache_root = cfg.cache_path()
    ea_flag = "ea" if cfg.preprocess.use_ea else "no_ea"
    ds = cfg.data.dataset.replace("/", "_")
    candidates = [
        p
        for p in cache_root.glob(f"{ds}_lr_{ea_flag}_*")
        if (p / "meta.json").exists() and (p / "holdout").exists()
    ]
    if not candidates:
        return None

    if cfg.data.subject_ids is not None:
        expected = cache_dir_for(cfg, len(cfg.data.subject_ids))
        if expected.exists():
            return expected
        target = set(cfg.data.subject_ids)
        for path in candidates:
            meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            cached = set(meta.get("dev_subjects", [])) | set(meta.get("test_subjects", []))
            if cached == target:
                return path

    return max(
        candidates,
        key=lambda p: json.loads((p / "meta.json").read_text(encoding="utf-8"))["n_subjects"],
    )


def load_holdout_arrays(cfg: ExperimentConfig) -> dict:
    """Load cached hold-out arrays; build cache if missing."""
    out_dir = _pick_holdout_cache_dir(cfg)
    if out_dir is None:
        out_dir = build_and_cache_holdout(cfg)
    holdout_dir = out_dir / "holdout"
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

    cfg.model.n_times = meta.get("model_n_times", meta["common_n_times"])
    cfg.model.n_channels = meta["n_channels"]

    return {
        "X_dev": np.load(holdout_dir / "X_dev.npy"),
        "y_dev": np.load(holdout_dir / "y_dev.npy"),
        "groups_dev": np.load(holdout_dir / "groups_dev.npy"),
        "X_test": np.load(holdout_dir / "X_test.npy"),
        "y_test": np.load(holdout_dir / "y_test.npy"),
        "groups_test": np.load(holdout_dir / "groups_test.npy"),
        "dev_subject_ids": np.load(holdout_dir / "dev_subject_ids.npy"),
        "test_subject_ids": np.load(holdout_dir / "test_subject_ids.npy"),
        "meta": meta,
        "cache_dir": out_dir,
    }


def load_full_cohort_arrays(cfg: ExperimentConfig, force: bool = False) -> dict:
    """Load all subjects as single arrays for LOSO (after same preprocess per fold)."""
    holdout = load_holdout_arrays(cfg)
    X = np.concatenate([holdout["X_dev"], holdout["X_test"]], axis=0)
    y = np.concatenate([holdout["y_dev"], holdout["y_test"]], axis=0)
    groups = np.concatenate([holdout["groups_dev"], holdout["groups_test"]], axis=0)
    return {"X": X, "y": y, "groups": groups, "meta": holdout["meta"]}
