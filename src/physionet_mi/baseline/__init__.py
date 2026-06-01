"""Classical baselines."""

from physionet_mi.baseline.csp_svm_classifier import run_csp_svm_holdout
from physionet_mi.baseline.lda_fbcsp import run_lda_holdout

__all__ = ["run_csp_svm_holdout", "run_lda_holdout"]
