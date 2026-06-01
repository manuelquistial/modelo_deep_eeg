"""CSP + SVM baseline — ported from modelo_bilstm (Sun et al. §2.6)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from physionet_mi.config import ExperimentConfig, load_config
from physionet_mi.data.cache import load_holdout_arrays
from physionet_mi.evaluation.reporting import save_run_artifacts
from physionet_mi.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class CSPSVMClassifier:
    """CSP → StandardScaler → SVM(RBF) with optional grid search."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        n_comp = config.get("n_components", 4)
        base_pipeline = Pipeline([
            ("csp", CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)),
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True)),
        ])
        if config.get("grid_search", True):
            param_grid = config.get("param_grid", {
                "svm__C": [0.1, 1, 10],
                "svm__gamma": ["scale", "auto", 0.01, 0.1],
                "svm__kernel": [config.get("svm_kernel", "rbf")],
            })
            cv = StratifiedKFold(
                n_splits=config.get("cv_folds", 3),
                shuffle=True,
                random_state=config.get("seed", 42),
            )
            self.model = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=config.get("n_jobs", -1),
            )
        else:
            self.model = base_pipeline
            self.model.set_params(
                svm__C=config.get("svm_C", 1.0),
                svm__gamma=config.get("svm_gamma", "scale"),
                svm__kernel=config.get("svm_kernel", "rbf"),
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CSPSVMClassifier":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def _csp_svm_config_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    cs = cfg.csp_svm
    return {
        "n_components": cs.n_components,
        "svm_kernel": cs.svm_kernel,
        "grid_search": cs.grid_search,
        "cv_folds": cs.cv_folds,
        "seed": cs.seed,
        "n_jobs": cs.n_jobs,
    }


def run_csp_svm_holdout(cfg: ExperimentConfig, run_name: str = "csp_svm_holdout") -> dict:
    """Train CSP+SVM on cached hold-out arrays (same deep_eeg preprocess as DL models)."""
    data = load_holdout_arrays(cfg)
    X_dev, y_dev = data["X_dev"], data["y_dev"]
    X_test, y_test = data["X_test"], data["y_test"]

    clf = CSPSVMClassifier(_csp_svm_config_dict(cfg))
    clf.fit(X_dev, y_dev)
    y_pred = clf.predict(X_test)

    out_dir = cfg.outputs_path() / run_name
    metrics = save_run_artifacts(
        out_dir,
        y_test,
        y_pred,
        {"model": "CSP+SVM", "use_ea": cfg.preprocess.use_ea},
        extra={"protocol": "holdout", "model": "CSP+SVM"},
    )
    logger.info("CSP+SVM baseline metrics: %s", metrics)
    return metrics


def main(config_path: str, run_name: str | None = None, project_root=None) -> None:
    from pathlib import Path

    setup_logging()
    config_path_p = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else config_path_p.parent.parent
    cfg = load_config(config_path_p, project_root=root)
    ea = "ea" if cfg.preprocess.use_ea else "no_ea"
    name = run_name or f"csp_svm_{ea}_holdout"
    run_csp_svm_holdout(cfg, name)
