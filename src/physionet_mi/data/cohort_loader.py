"""Unified cohort loading for all supported datasets."""

from __future__ import annotations

from physionet_mi.config import ExperimentConfig
from physionet_mi.constants import SUPPORTED_DATASETS
from physionet_mi.data.bnci_loader import load_bnci2014_001_cohort
from physionet_mi.data.moabb_loader import SubjectRecord, load_physionet_cohort


def load_cohort(cfg: ExperimentConfig) -> tuple[dict[int, SubjectRecord], list[str]]:
    name = cfg.data.dataset.lower()
    if name == "physionet":
        return load_physionet_cohort(cfg)
    if name == "bnci2014_001":
        return load_bnci2014_001_cohort(cfg)
    raise ValueError(f"Unknown data.dataset={cfg.data.dataset!r}. Use {SUPPORTED_DATASETS}")
