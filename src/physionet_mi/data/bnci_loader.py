"""MOABB BNCI2014_001 loader — aligned with modelo_bilstm import."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from physionet_mi.config import ExperimentConfig
from physionet_mi.constants import BINARY_EVENTS, LABEL_NAME_TO_ID
from physionet_mi.data.moabb_loader import SubjectRecord, setup_mne_paths

logger = logging.getLogger(__name__)

BNCI2014_001_CHANNEL_NAMES: list[str] = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]

EVENT_CODE_TO_NAME: dict[int, str] = {
    1: "left_hand",
    2: "right_hand",
    3: "feet",
    4: "tongue",
}


def _normalize_event_label(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        name = EVENT_CODE_TO_NAME.get(int(value))
        if name:
            return name
        return str(int(value))
    s = str(value).strip().lower().replace(" ", "_")
    aliases = {"left": "left_hand", "right": "right_hand"}
    return aliases.get(s, s)


def _encode_labels(y: Any, event_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y_raw = np.asarray(y)
    name_to_id = {name: i for i, name in enumerate(event_names)}
    encoded = np.full(len(y_raw), -1, dtype=np.int64)
    for i, val in enumerate(y_raw):
        canonical = _normalize_event_label(val)
        if canonical in name_to_id:
            encoded[i] = name_to_id[canonical]
    keep = encoded >= 0
    return encoded[keep].astype(np.int64), keep


def _class_to_workshop(y_class: np.ndarray) -> np.ndarray:
    """Map MOABB class 0|1 to workshop labels 1|2 used internally."""
    out = np.empty(len(y_class), dtype=np.int64)
    for i, c in enumerate(y_class):
        out[i] = LABEL_NAME_TO_ID["left_hand"] if int(c) == 0 else LABEL_NAME_TO_ID["right_hand"]
    return out


def load_bnci2014_001_cohort(cfg: ExperimentConfig) -> tuple[dict[int, SubjectRecord], list[str]]:
    """Load BNCI2014_001 binary L/R hand into SubjectRecord dict."""
    setup_mne_paths(cfg.mne_data_path())
    dataset = BNCI2014_001()

    if cfg.data.subject_ids is not None:
        subject_ids = [int(s) for s in cfg.data.subject_ids]
    else:
        subject_ids = [int(s) for s in dataset.subject_list]

    event_names = list(BINARY_EVENTS)
    paradigm_kw: dict[str, Any] = {
        "n_classes": cfg.data.bnci_n_classes,
        "fmin": cfg.data.bnci_fmin,
        "fmax": cfg.data.bnci_fmax,
        "tmin": cfg.data.bnci_tmin,
        "tmax": cfg.data.bnci_tmax,
        "resample": cfg.data.bnci_resample,
    }
    try:
        paradigm = MotorImagery(events=event_names, **paradigm_kw)
    except TypeError:
        logger.warning("MOABB MotorImagery without events=; filtering labels manually.")
        paradigm = MotorImagery(**paradigm_kw)

    ch_names = list(BNCI2014_001_CHANNEL_NAMES)
    subj_data: dict[int, SubjectRecord] = {}

    for sid in subject_ids:
        logger.info("BNCI2014_001 subject %s …", sid)
        X, y, meta = paradigm.get_data(dataset, subjects=[int(sid)])
        if X.ndim != 3:
            raise ValueError(f"Unexpected shape for subject {sid}: {X.shape}")
        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

        y_class, keep = _encode_labels(y, event_names)
        if not np.all(keep):
            X = X[keep]
            if hasattr(meta, "iloc"):
                meta = meta.loc[keep].reset_index(drop=True)
        if len(y_class) == 0:
            logger.warning("Subject %s: no left/right trials.", sid)
            continue

        y_workshop = _class_to_workshop(y_class)
        n_epochs = len(y_class)
        trials_s = [X[i].astype(np.float32) for i in range(n_epochs)]

        if "session" in meta.columns:
            session_col = meta["session"].astype(str).values
        else:
            session_col = np.array(["session_0"] * n_epochs, dtype=object)
        run_col = meta["run"].values if "run" in meta.columns else np.zeros(n_epochs, dtype=int)

        run_ids_s = np.zeros(n_epochs, dtype=np.int64)
        seen: dict = {}
        rid = 0
        for i, key in enumerate(zip(session_col, run_col, strict=True)):
            if key not in seen:
                seen[key] = rid
                rid += 1
            run_ids_s[i] = seen[key]

        n_ch = trials_s[0].shape[1]
        if n_ch != len(ch_names):
            ch_names = [f"Ch{i}" for i in range(n_ch)]

        subj_data[int(sid)] = {
            "trials": trials_s,
            "labels": y_workshop,
            "run_ids": run_ids_s,
            "label_map": LABEL_NAME_TO_ID.copy(),
            "label_names": list(BINARY_EVENTS),
            "original_labels_seq": y_class,
            "ch_names": list(ch_names),
            "n_epochs": n_epochs,
            "n_channels": n_ch,
            "n_times": trials_s[0].shape[0],
        }

    if not subj_data:
        raise RuntimeError("No BNCI2014_001 subjects loaded.")
    logger.info("Loaded BNCI2014_001: %d subjects.", len(subj_data))
    return subj_data, ch_names
