"""Model factory — dispatch by cfg.model.name."""

from __future__ import annotations

import torch.nn as nn

from physionet_mi.config import ExperimentConfig
from physionet_mi.models.eegme import build_eegme
from physionet_mi.models.eegnet import build_eegnet

SUPPORTED_MODELS = ("eegme", "eegnet")


def build_model(cfg: ExperimentConfig) -> nn.Module:
    """Build neural model from experiment config."""
    name = cfg.model.name.lower()
    if name == "eegme":
        return build_eegme(cfg)
    if name == "eegnet":
        return build_eegnet(cfg)
    raise ValueError(f"Unknown model.name={cfg.model.name!r}. Choose from {SUPPORTED_MODELS}")


def default_run_prefix(cfg: ExperimentConfig) -> str:
    """Run name prefix, e.g. eegme_no_ea or eegnet_ea."""
    ea = "ea" if cfg.preprocess.use_ea else "no_ea"
    return f"{cfg.model.name.lower()}_{ea}"
