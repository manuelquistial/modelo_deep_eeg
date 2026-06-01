"""Training pipelines."""

from physionet_mi.training.holdout import run_holdout
from physionet_mi.training.loso import run_loso

__all__ = ["run_holdout", "run_loso"]
