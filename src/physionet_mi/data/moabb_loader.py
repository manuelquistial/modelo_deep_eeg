"""MOABB PhysioNet MI data loading."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import mne
import numpy as np
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

from physionet_mi.config import ExperimentConfig
from physionet_mi.constants import BINARY_EVENTS, LABEL_NAME_TO_ID

logger = logging.getLogger(__name__)

SubjectRecord = dict[str, Any]

# PhysioNet MI: subject 88 was recorded at 128 Hz (incompatible with 160 Hz cohort).
EXCLUDED_SUBJECT_IDS = {88}


def setup_mne_paths(data_dir: str | Path) -> Path:
    """Configure MNE/MOABB data directories (overrides user-global config)."""
    resolved = Path(data_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)

    # Environment variables take precedence in MNE/MOABB; set explicitly.
    os.environ["MNE_DATA"] = str(resolved)
    os.environ["MOABB_DATA"] = str(resolved)
    # PhysioNet MI uses the EEGBCI signifier in MOABB/MNE download helpers.
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = str(resolved)

    mne.set_config("MNE_DATA", str(resolved), set_env=True)
    mne.set_config("MOABB_DATA", str(resolved), set_env=True)
    mne.set_config("MNE_DATASETS_EEGBCI_PATH", str(resolved), set_env=True)

    logger.info("MNE_DATA=%s", mne.get_config("MNE_DATA"))
    logger.info("MNE_DATASETS_EEGBCI_PATH=%s", mne.get_config("MNE_DATASETS_EEGBCI_PATH"))
    return resolved


def get_physionet_dataset():
    return PhysionetMI()


def get_motor_imagery_paradigm(events: list[str] | None = None) -> MotorImagery:
    events = events or list(BINARY_EVENTS)
    return MotorImagery(events=events, n_classes=2)


def get_real_eeg_ch_names(dataset: PhysionetMI, subject: int = 1) -> list[str]:
    data = dataset.get_data(subjects=[subject])
    subject_key = next(iter(data.keys()))
    session_key = next(iter(data[subject_key].keys()))
    run_key = next(iter(data[subject_key][session_key].keys()))
    raw = data[subject_key][session_key][run_key]
    return [ch for ch in raw.info["ch_names"] if ch.upper() != "STIM"]


def labels_to_binary_int(labels_arr: np.ndarray) -> np.ndarray:
    labels_arr = np.asarray(labels_arr).astype(str)
    return np.array([LABEL_NAME_TO_ID[str(lbl)] for lbl in labels_arr], dtype=int)


def build_subject_data(
    dataset: PhysionetMI,
    paradigm: MotorImagery,
    subject_ids: list[int],
    ch_names: list[str],
) -> dict[int, SubjectRecord]:
    """Load per-subject trials for binary left/right MI."""
    subj_data: dict[int, SubjectRecord] = {}

    for sid in subject_ids:
        logger.info("Processing subject %s...", sid)
        try:
            X_s, y_s, metadata_s = paradigm.get_data(dataset=dataset, subjects=[sid])
        except Exception as e:
            logger.warning("Skipping subject %s: %s", sid, e)
            continue

        if X_s.ndim != 3 or X_s.shape[0] == 0:
            logger.warning("Subject %s: no valid epochs.", sid)
            continue

        y_s = np.asarray(y_s).astype(str)
        valid_mask = np.isin(y_s, BINARY_EVENTS)
        X_s = X_s[valid_mask]
        y_s = y_s[valid_mask]
        metadata_s = metadata_s.iloc[np.where(valid_mask)[0]].reset_index(drop=True)

        if X_s.shape[0] == 0:
            logger.warning("Subject %s: no left/right epochs.", sid)
            continue

        n_epochs, n_ch, n_times = X_s.shape
        if n_ch != len(ch_names):
            raise ValueError(
                f"Subject {sid}: X has {n_ch} channels; expected {len(ch_names)}."
            )

        trials_s = [X_s[i].T.astype(np.float32) for i in range(n_epochs)]
        labels_s = labels_to_binary_int(y_s)

        run_col_s = metadata_s["run"].values if "run" in metadata_s.columns else np.zeros(n_epochs, dtype=int)
        session_col_s = (
            np.asarray(metadata_s["session"].values, dtype=object)
            if "session" in metadata_s.columns
            else np.array(["session_0"] * n_epochs, dtype=object)
        )

        run_ids_s = np.zeros(n_epochs, dtype=np.int64)
        seen_s: dict = {}
        current_run_id = 0
        for i, key in enumerate(zip(session_col_s, run_col_s, strict=True)):
            if key not in seen_s:
                seen_s[key] = current_run_id
                current_run_id += 1
            run_ids_s[i] = seen_s[key]

        subj_data[sid] = {
            "trials": trials_s,
            "labels": labels_s,
            "run_ids": run_ids_s,
            "label_map": LABEL_NAME_TO_ID.copy(),
            "label_names": list(BINARY_EVENTS),
            "original_labels_seq": y_s,
            "ch_names": list(ch_names),
            "n_epochs": n_epochs,
            "n_channels": n_ch,
            "n_times": n_times,
        }

    return subj_data


def _resolve_subject_ids(dataset: PhysionetMI, cfg: ExperimentConfig) -> list[int]:
    if cfg.data.subject_ids is not None:
        subject_ids = [int(s) for s in cfg.data.subject_ids]
    else:
        subject_ids = [int(s) for s in dataset.subject_list]
    subject_ids = [s for s in subject_ids if s not in EXCLUDED_SUBJECT_IDS]
    if not subject_ids:
        raise ValueError("No subjects left after excluding incompatible IDs.")
    return subject_ids


def _download_subjects_with_retry(
    dataset: PhysionetMI,
    subject_ids: list[int],
    data_dir: Path,
    *,
    max_retries: int = 8,
    base_delay_s: float = 5.0,
) -> None:
    """Download each subject separately so partial progress survives network errors."""
    pending = list(subject_ids)
    for attempt in range(1, max_retries + 1):
        failed: list[int] = []
        for sid in pending:
            try:
                dataset.download(
                    subject_list=[sid],
                    path=str(data_dir),
                    accept=True,
                    verbose=False,
                )
            except Exception as exc:
                logger.warning("Download failed for subject %s (attempt %d): %s", sid, attempt, exc)
                failed.append(sid)
        if not failed:
            return
        pending = failed
        delay = base_delay_s * (2 ** (attempt - 1))
        logger.warning(
            "%d subjects still missing; retrying in %.0fs (attempt %d/%d)...",
            len(pending),
            delay,
            attempt,
            max_retries,
        )
        time.sleep(delay)
    raise RuntimeError(
        f"Could not download all subjects after {max_retries} attempts. "
        f"Still missing: {pending}"
    )


def load_physionet_cohort(cfg: ExperimentConfig) -> tuple[dict[int, SubjectRecord], list[str]]:
    """Download/load cohort and return subject dict + channel names."""
    data_dir = setup_mne_paths(cfg.mne_data_path())

    dataset = get_physionet_dataset()
    subject_ids = _resolve_subject_ids(dataset, cfg)

    logger.info("Downloading PhysioNet MI if missing (%d subjects)...", len(subject_ids))
    _download_subjects_with_retry(dataset, subject_ids, data_dir)

    paradigm = get_motor_imagery_paradigm(cfg.data.binary_events)
    ch_names = get_real_eeg_ch_names(dataset, subject=subject_ids[0])

    subj_data = build_subject_data(dataset, paradigm, subject_ids, ch_names)
    if not subj_data:
        raise RuntimeError("No subject data loaded. Check download path and network access.")
    logger.info("Loaded %d subjects.", len(subj_data))
    return subj_data, ch_names
