"""Load experiment configuration from YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from physionet_mi.constants import BINARY_EVENTS, DEFAULT_FREQ_BANDS, SFREQ_PHYSIONET


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path, project_root: Path | None = None) -> "ExperimentConfig":
    """Load config with optional `defaults: other.yaml` inheritance."""
    config_path = Path(config_path).resolve()
    root = project_root or config_path.parent.parent
    raw = _load_yaml(config_path)

    defaults_name = raw.pop("defaults", None)
    if defaults_name:
        defaults_path = config_path.parent / defaults_name
        if not defaults_path.exists():
            defaults_path = root / "configs" / defaults_name
        base = _load_yaml(defaults_path)
        raw = _deep_merge(base, raw)

    env_data_dir = os.environ.get("PHYSIONET_MI_DATA_DIR")
    if env_data_dir:
        raw.setdefault("data", {})["mne_data_dir"] = env_data_dir

    return ExperimentConfig.from_dict(raw, root=root)


@dataclass
class DataConfig:
    dataset: str = "physionet"  # physionet | bnci2014_001
    mne_data_dir: str = "data/mne"
    subject_ids: list[int] | None = None
    binary_events: list[str] = field(default_factory=lambda: list(BINARY_EVENTS))
    cache_dir: str = "data/processed"
    # BNCI2014_001 (MOABB) — same defaults as modelo_bilstm
    bnci_tmin: float = 0.0
    bnci_tmax: float = 4.0
    bnci_resample: float = 125.0
    bnci_fmin: float = 1.0
    bnci_fmax: float = 40.0
    bnci_n_classes: int = 2


@dataclass
class PreprocessConfig:
    use_ea: bool = False
    highpass_hz: float = 4.0
    highpass_order: int = 4
    outlier_uv: float = 800.0
    auto_detect_outlier_units: bool = True
    crop_mode: str = "crop"
    ea_reg: float = 1e-10
    ea_verify_enabled: bool = True


@dataclass
class SplitConfig:
    test_size: float = 0.20
    random_state: int = 42
    val_ratio: float = 0.15


@dataclass
class ModelConfig:
    name: str = "eegme"  # eegme | eegnet
    n_channels: int = 64
    n_times: int = 480
    f1: int = 7
    embed_dim: int = 32
    num_classes: int = 2
    dropout: float = 0.25
    # EEGNet (Lawhern et al. 2018 — ported from modelo_bilstm)
    eegnet_F1: int = 8
    eegnet_D: int = 2
    eegnet_F2: int = 16
    eegnet_kernel_length: int = 64
    eegnet_dropout: float = 0.5


@dataclass
class CspSvmConfig:
    n_components: int = 4
    svm_kernel: str = "rbf"
    grid_search: bool = True
    cv_folds: int = 3
    seed: int = 42
    n_jobs: int = -1


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 1e-3
    max_epochs: int = 100
    early_stopping_patience: int = 20
    scheduler_patience: int = 10
    scheduler_factor: float = 0.1
    seed: int = 42
    device: str = "auto"
    trial_wise_normalize: bool = True
    use_triplet: bool = False


@dataclass
class BaselineConfig:
    freq_bands: list[tuple[int, int]] = field(default_factory=lambda: list(DEFAULT_FREQ_BANDS))
    bandpass_order: int = 5
    n_csp_components: int = 4
    fbcsp_reg: str | float = "ledoit_wolf"
    fbcsp_norm_trace: bool = True
    feature_selection_k: int = 16
    use_feature_selection: bool = True


@dataclass
class EvalConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "balanced_accuracy", "macro_f1", "kappa"]
    )


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    csp_svm: CspSvmConfig = field(default_factory=CspSvmConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    outputs_dir: str = "outputs"
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path | None = None) -> "ExperimentConfig":
        root = root or Path.cwd()

        data_raw = raw.get("data", {})
        preprocess_raw = raw.get("preprocess", {})
        split_raw = raw.get("split", {})
        model_raw = raw.get("model", {})
        train_raw = raw.get("train", {})
        baseline_raw = raw.get("baseline", {})
        csp_svm_raw = raw.get("csp_svm", {})
        eval_raw = raw.get("eval", {})

        bands = baseline_raw.get("freq_bands", DEFAULT_FREQ_BANDS)
        bands_tuples = [tuple(b) for b in bands]

        return cls(
            data=DataConfig(**{**DataConfig().__dict__, **data_raw}),
            preprocess=PreprocessConfig(**{**PreprocessConfig().__dict__, **preprocess_raw}),
            split=SplitConfig(**{**SplitConfig().__dict__, **split_raw}),
            model=ModelConfig(**{**ModelConfig().__dict__, **model_raw}),
            train=TrainConfig(**{**TrainConfig().__dict__, **train_raw}),
            baseline=BaselineConfig(
                **{
                    **BaselineConfig().__dict__,
                    **{k: v for k, v in baseline_raw.items() if k != "freq_bands"},
                    "freq_bands": bands_tuples,
                }
            ),
            csp_svm=CspSvmConfig(**{**CspSvmConfig().__dict__, **csp_svm_raw}),
            eval=EvalConfig(**{**EvalConfig().__dict__, **eval_raw}),
            outputs_dir=raw.get("outputs_dir", "outputs"),
            project_root=root,
        )

    def resolve_path(self, relative: str) -> Path:
        p = Path(relative)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def mne_data_path(self) -> Path:
        return self.resolve_path(self.data.mne_data_dir)

    def cache_path(self) -> Path:
        return self.resolve_path(self.data.cache_dir)

    def outputs_path(self) -> Path:
        return self.resolve_path(self.outputs_dir)


def dataset_sfreq(cfg: ExperimentConfig) -> float:
    """Sampling rate for the active dataset (Hz)."""
    if cfg.data.dataset == "bnci2014_001":
        return float(cfg.data.bnci_resample)
    return float(SFREQ_PHYSIONET)
