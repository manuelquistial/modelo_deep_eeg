"""FBCSP + LDA baseline (workshop pipeline on cached arrays)."""

from __future__ import annotations

import logging

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

from physionet_mi.config import ExperimentConfig, load_config, dataset_sfreq
from physionet_mi.constants import class_to_workshop_label
from physionet_mi.data.cache import load_holdout_arrays
from physionet_mi.evaluation.reporting import save_run_artifacts
from physionet_mi.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class BinaryFBCSP:
    def __init__(
        self,
        freq_bands,
        sfreq: float,
        n_components: int = 4,
        filter_order: int = 5,
        reg="ledoit_wolf",
        norm_trace: bool = True,
    ):
        self.freq_bands = freq_bands
        self.sfreq = sfreq
        self.n_components = n_components
        self.filter_order = filter_order
        self.reg = reg
        self.norm_trace = norm_trace
        self.csps_ = []
        self.feature_names_ = []

    def _bandpass(self, X: np.ndarray, band) -> np.ndarray:
        low, high = band
        sos = butter(self.filter_order, [low, high], btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, X, axis=2).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.csps_ = []
        self.feature_names_ = []
        y_workshop = np.array([class_to_workshop_label(int(c)) for c in y])
        for band in self.freq_bands:
            Xb = self._bandpass(X, band)
            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=True,
                norm_trace=self.norm_trace,
                transform_into="average_power",
            )
            csp.fit(Xb, y_workshop)
            self.csps_.append(csp)
            for comp_idx in range(self.n_components):
                self.feature_names_.append(f"band_{band[0]}_{band[1]}Hz_csp{comp_idx+1}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for band, csp in zip(self.freq_bands, self.csps_):
            Xb = self._bandpass(X, band)
            feats.append(csp.transform(Xb))
        return np.concatenate(feats, axis=1).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


def extract_fbcsp_features(X_dev, y_dev, X_test, cfg: ExperimentConfig):

    bl = cfg.baseline
    fbcsp = BinaryFBCSP(
        freq_bands=bl.freq_bands,
        sfreq=dataset_sfreq(cfg),
        n_components=bl.n_csp_components,
        filter_order=bl.bandpass_order,
        reg=bl.fbcsp_reg,
        norm_trace=bl.fbcsp_norm_trace,
    )
    X_dev_f = fbcsp.fit_transform(X_dev, y_dev)
    X_test_f = fbcsp.transform(X_test)

    if bl.use_feature_selection:
        y_w = np.array([class_to_workshop_label(int(c)) for c in y_dev])
        k = min(bl.feature_selection_k, X_dev_f.shape[1])
        selector = SelectKBest(
            score_func=lambda X, y: mutual_info_classif(
                X, y, discrete_features=False, random_state=cfg.split.random_state
            ),
            k=k,
        )
        X_dev_f = selector.fit_transform(X_dev_f, y_w)
        X_test_f = selector.transform(X_test_f)
    else:
        selector = None

    scaler = StandardScaler()
    X_dev_f = scaler.fit_transform(X_dev_f).astype(np.float32)
    X_test_f = scaler.transform(X_test_f).astype(np.float32)
    return X_dev_f, X_test_f


def run_lda_holdout(cfg: ExperimentConfig, run_name: str = "lda_holdout") -> dict:
    data = load_holdout_arrays(cfg)
    X_dev, y_dev = data["X_dev"], data["y_dev"]
    X_test, y_test = data["X_test"], data["y_test"]

    X_dev_f, X_test_f = extract_fbcsp_features(X_dev, y_dev, X_test, cfg)

    y_dev_w = np.array([class_to_workshop_label(int(c)) for c in y_dev])
    y_test_w = np.array([class_to_workshop_label(int(c)) for c in y_test])

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
    lda.fit(X_dev_f, y_dev_w)
    y_pred_w = lda.predict(X_test_f)
    y_pred = (y_pred_w - 1).astype(np.int64)  # workshop 1|2 -> class 0|1

    out_dir = cfg.outputs_path() / run_name
    metrics = save_run_artifacts(
        out_dir,
        y_test,
        y_pred,
        {"model": "FBCSP+LDA", "use_ea": cfg.preprocess.use_ea},
        extra={"protocol": "holdout", "model": "FBCSP+LDA"},
    )
    logger.info("LDA baseline metrics: %s", metrics)
    return metrics


def main(config_path: str, run_name: str | None = None, project_root=None) -> None:
    from pathlib import Path

    setup_logging()
    config_path_p = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else config_path_p.parent.parent
    cfg = load_config(config_path_p, project_root=root)
    name = run_name or f"lda_{'ea' if cfg.preprocess.use_ea else 'no_ea'}_holdout"
    run_lda_holdout(cfg, name)
