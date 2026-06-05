"""Riemannian geometry baselines for MI classification."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from physionet_mi.data.preprocessing import trial_covariance_ea

logger = logging.getLogger(__name__)

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangent_space import TangentSpace
    from pyriemann.classification import MDM

    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False


def _trial_covariances(X: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """(N, C, T) -> (N, C, C) covariance matrices."""
    return np.stack([trial_covariance_ea(X[i], reg=reg) for i in range(len(X))], axis=0)


def _logeuclidean_features(covs: np.ndarray) -> np.ndarray:
    """Vectorize log-transformed covariances (fallback without pyriemann)."""
    feats = []
    for C in covs:
        w, v = np.linalg.eigh(C)
        w = np.maximum(w, 1e-12)
        log_c = (v * np.log(w)) @ v.T
        triu_idx = np.triu_indices(log_c.shape[0])
        feats.append(log_c[triu_idx])
    return np.stack(feats, axis=0).astype(np.float64)


class RiemannMDMClassifier:
    """Minimum Distance to Riemannian Mean classifier."""

    def __init__(self, estimator: str = "lwf") -> None:
        self.estimator = estimator
        self._mdm = None
        self._use_pyriemann = HAS_PYRIEMANN

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiemannMDMClassifier":
        if self._use_pyriemann:
            cov = Covariances(estimator=self.estimator).fit_transform(X)
            self._mdm = MDM(metric="riemann")
            self._mdm.fit(cov, y)
        else:
            cov = _trial_covariances(X)
            self._cov_train = cov
            self._y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._use_pyriemann and self._mdm is not None:
            cov = Covariances(estimator=self.estimator).fit_transform(X)
            return self._mdm.predict(cov)
        cov = _trial_covariances(X)
        preds = []
        for c in cov:
            dists = [np.linalg.norm(c - r, "fro") for r in self._cov_train]
            preds.append(self._y_train[int(np.argmin(dists))])
        return np.asarray(preds, dtype=int)


class RiemannTangentLRClassifier:
    """Tangent space projection + logistic regression."""

    def __init__(self, estimator: str = "lwf", max_iter: int = 2000) -> None:
        self.estimator = estimator
        self.max_iter = max_iter
        self._ts = None
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=42,
        )
        self._use_pyriemann = HAS_PYRIEMANN

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiemannTangentLRClassifier":
        if self._use_pyriemann:
            cov = Covariances(estimator=self.estimator).fit_transform(X)
            self._ts = TangentSpace(metric="riemann")
            feats = self._ts.fit_transform(cov)
        else:
            cov = _trial_covariances(X)
            feats = _logeuclidean_features(cov)
        feats = self._scaler.fit_transform(feats)
        self._clf.fit(feats, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._use_pyriemann and self._ts is not None:
            cov = Covariances(estimator=self.estimator).fit_transform(X)
            feats = self._ts.transform(cov)
        else:
            cov = _trial_covariances(X)
            feats = _logeuclidean_features(cov)
        feats = self._scaler.transform(feats)
        return self._clf.predict(feats)


def run_riemannian_model(
    model_name: str,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Train and predict; returns y_pred, train_time_seconds, extra info."""
    t0 = time.perf_counter()
    if model_name == "riemann_mdm":
        clf = RiemannMDMClassifier()
    elif model_name in ("riemann_ts_lr", "riemann_fgmdm"):
        clf = RiemannTangentLRClassifier()
    else:
        raise ValueError(f"Unknown Riemannian model: {model_name}")
    clf.fit(X_dev, y_dev)
    y_pred = clf.predict(X_test)
    elapsed = time.perf_counter() - t0
    return y_pred, elapsed, {"pyriemann": HAS_PYRIEMANN, "model": model_name}
