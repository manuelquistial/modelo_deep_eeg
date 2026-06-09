"""Mu-band spatial topomaps from cached EEG arrays (MNE)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from physionet_mi.analysis.erd_ers import MU_BAND, _bandpower
from physionet_mi.constants import CLASS_TO_NAME
from physionet_mi.data.bnci_loader import BNCI2014_001_CHANNEL_NAMES
from physionet_mi.paper.figure_common import save_figure

MU_BAND_LABEL = "8–13 Hz mu-band power topography"
NOT_ERD_ERS = "Not baseline-corrected ERD/ERS"


def _pick_dataset_cache(cache_root: Path, dataset: str, *, use_ea: bool = False) -> Path | None:
    """Return the largest hold-out cache for a dataset and EA flag."""
    ea_flag = "ea" if use_ea else "no_ea"
    ds = dataset.replace("/", "_")
    candidates = [
        path
        for path in cache_root.glob(f"{ds}_lr_{ea_flag}_*")
        if (path / "meta.json").exists() and (path / "holdout").exists()
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: json.loads((path / "meta.json").read_text(encoding="utf-8"))["n_subjects"],
    )


def _load_cache_arrays(cache_dir: Path | None) -> dict | None:
    if cache_dir is None:
        return None
    holdout = cache_dir / "holdout"
    meta_path = cache_dir / "meta.json"
    if not holdout.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    X = np.concatenate([
        np.load(holdout / "X_dev.npy"),
        np.load(holdout / "X_test.npy"),
    ])
    y = np.concatenate([
        np.load(holdout / "y_dev.npy"),
        np.load(holdout / "y_test.npy"),
    ])
    return {"X": X, "y": y, "meta": meta}


def _mean_mu_topography(X: np.ndarray, y: np.ndarray, label_idx: int, sfreq: float) -> np.ndarray:
    mask = y == label_idx
    if not mask.any():
        return np.full(X.shape[1], np.nan)
    trials = X[mask]
    powers = []
    for ch in range(trials.shape[1]):
        vals = [_bandpower(tr[ch], sfreq, MU_BAND) for tr in trials]
        powers.append(float(np.nanmean(vals)))
    return np.asarray(powers)


def _make_info(ch_names: list[str], sfreq: float) -> mne.Info:
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")
    return info


def _adjust_topo_layout(fig, *, top: float = 0.90) -> None:
    """Manual spacing; MNE topomap axes are not tight_layout-compatible."""
    fig.subplots_adjust(left=0.06, right=0.90, top=top, bottom=0.08, wspace=0.20, hspace=0.30)


def _plot_topo_panel(ax, values: np.ndarray, info: mne.Info, title: str, vmin, vmax):
    kwargs = dict(
        axes=ax,
        show=False,
        contours=0,
        sensors=True,
        extrapolate="head",
        image_interp="cubic",
    )
    if vmin is not None or vmax is not None:
        kwargs["vlim"] = (vmin, vmax)
    im, _ = mne.viz.plot_topomap(values, info, **kwargs)
    ax.set_title(title, fontsize=10)
    return im


def _bnci_sparse_topography(
    trial_csv: Path,
    label: str,
    use_ea: bool,
    ch_names: list[str],
) -> np.ndarray:
    """Motor-channel mu power from neurophysiology CSV when full cache unavailable."""
    df = pd.read_csv(trial_csv)
    sub = df[(df["label"] == label) & (df["use_ea"] == use_ea)]
    alias_map = {
        "C3": "mu_power_C3",
        "C4": "mu_power_C4",
    }
    values = np.full(len(ch_names), np.nan)
    ch_index = {n: i for i, n in enumerate(ch_names)}
    for ch, col in alias_map.items():
        if ch in ch_index and col in sub.columns:
            values[ch_index[ch]] = float(sub[col].mean())
    return values


def generate_topographic_figures(
    artifacts_root: Path,
    output_stem: Path,
    ea_stem: Path | None = None,
) -> dict[str, tuple[Path, Path] | None]:
    cache_root = artifacts_root / "cache"
    neuro_root = artifacts_root / "runs" / "publishable" / "neurophysiology"
    out: dict[str, tuple[Path, Path] | None] = {"motor_imagery": None, "before_after_ea": None}

    phys_data = _load_cache_arrays(_pick_dataset_cache(cache_root, "physionet", use_ea=False))
    bnci_trial = neuro_root / "bnci" / "erd_ers_trial_level.csv"

    if phys_data is None:
        return out

    ch_phys = phys_data["meta"]["ch_names"]
    sfreq_phys = float(phys_data["meta"].get("sfreq", 160))
    info_phys = _make_info(ch_phys, sfreq_phys)

    panels = []
    panel_infos = []
    for label_idx, label_name in [(0, "left_hand"), (1, "right_hand")]:
        vals = _mean_mu_topography(phys_data["X"], phys_data["y"], label_idx, sfreq_phys)
        panels.append((vals, f"PhysioNet — {label_name.replace('_', ' ')}"))
        panel_infos.append(info_phys)

    bnci_data = _load_cache_arrays(_pick_dataset_cache(cache_root, "bnci2014_001", use_ea=False))
    if bnci_data is not None:
        ch_bnci = bnci_data["meta"]["ch_names"]
        sfreq_bnci = float(bnci_data["meta"].get("sfreq", 125.0))
        info_bnci = _make_info(ch_bnci, sfreq_bnci)
        for label_idx, label_name in [(0, "left_hand"), (1, "right_hand")]:
            vals = _mean_mu_topography(bnci_data["X"], bnci_data["y"], label_idx, sfreq_bnci)
            panels.append((vals, f"BNCI — {label_name.replace('_', ' ')}"))
            panel_infos.append(info_bnci)
    elif bnci_trial.exists():
        ch_bnci = list(BNCI2014_001_CHANNEL_NAMES)
        info_bnci = _make_info(ch_bnci, 125.0)
        for label in ("left_hand", "right_hand"):
            vals = _bnci_sparse_topography(bnci_trial, label, use_ea=True, ch_names=ch_bnci)
            panels.append((vals, f"BNCI — {label.replace('_', ' ')}"))
            panel_infos.append(info_bnci)

    all_finite = [v for v, _ in panels for v in v[np.isfinite(v)]]
    vmin, vmax = np.percentile(all_finite, [5, 95]) if all_finite else (None, None)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ims = []
    for ax, (vals, title), info in zip(axes.ravel(), panels, panel_infos):
        im = _plot_topo_panel(ax, vals, info, title, vmin, vmax)
        ims.append(im)

    fig.suptitle(f"{MU_BAND_LABEL}\n({NOT_ERD_ERS})", fontsize=11, y=0.98)
    if ims:
        cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
        cbar.set_label("Mu-band power (a.u.)")
    _adjust_topo_layout(fig, top=0.86)
    out["motor_imagery"] = save_figure(fig, output_stem)
    plt.close(fig)

    # Before vs after EA (PhysioNet full scalp; BNCI sparse if CSV available)
    phys_no = _load_cache_arrays(_pick_dataset_cache(cache_root, "physionet", use_ea=False))
    phys_ea = _load_cache_arrays(_pick_dataset_cache(cache_root, "physionet", use_ea=True))
    if phys_no and phys_ea and ea_stem:
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 7))
        ea_panels = [
            (_mean_mu_topography(phys_no["X"], phys_no["y"], 0, sfreq_phys), "PhysioNet left — before EA"),
            (_mean_mu_topography(phys_ea["X"], phys_ea["y"], 0, sfreq_phys), "PhysioNet left — after EA"),
            (_mean_mu_topography(phys_no["X"], phys_no["y"], 1, sfreq_phys), "PhysioNet right — before EA"),
            (_mean_mu_topography(phys_ea["X"], phys_ea["y"], 1, sfreq_phys), "PhysioNet right — after EA"),
        ]
        ea_vals = [v for v, _ in ea_panels for v in v[np.isfinite(v)]]
        evmin, evmax = np.percentile(ea_vals, [5, 95]) if ea_vals else (vmin, vmax)
        ims2 = []
        for ax, (vals, title) in zip(axes2.ravel(), ea_panels):
            ims2.append(_plot_topo_panel(ax, vals, info_phys, title, evmin, evmax))
        fig2.suptitle(f"EA effect on {MU_BAND_LABEL} (PhysioNet)", fontsize=11, y=0.98)
        if ims2:
            fig2.colorbar(ims2[0], ax=axes2.ravel().tolist(), fraction=0.025, pad=0.04)
        _adjust_topo_layout(fig2, top=0.90)
        out["before_after_ea"] = save_figure(fig2, ea_stem)
        plt.close(fig2)

    return out


def write_topoplot_notes(path: Path, *, bnci_sparse: bool) -> None:
    bnci_note = (
        "BNCI panels use motor-channel mu power (C3, C4) from `erd_ers_trial_level.csv` "
        "with MNE head interpolation; full 22-channel cache was not available locally.\n"
        if bnci_sparse else
        "BNCI panels computed from cached hold-out arrays.\n"
    )
    path.write_text(
        "# fig_topoplots — methodological notes\n\n"
        f"## Terminology\n- Maps are **{MU_BAND_LABEL}**.\n"
        f"- **{NOT_ERD_ERS}** — no baseline/rest reference window was applied.\n\n"
        "## Parameters\n"
        "- Frequency band: 8–13 Hz (mu), Welch PSD (`scipy.signal.welch`).\n"
        "- Time window: full cached trial epoch after preprocessing/cropping.\n"
        "- Averaging: mean across all trials of the imagined class (DEV+TEST subjects).\n"
        "- PhysioNet: 64 channels, standard_1020 montage (`on_missing='ignore'`).\n"
        f"- {bnci_note}"
        "- Before/after EA: PhysioNet caches `physionet_lr_no_ea_*` vs `physionet_lr_ea_*`.\n\n"
        "## Limitations\n"
        "- Topomaps reflect band power, not event-related desynchronization percentage.\n"
        "- BNCI sparse interpolation should be interpreted as illustrative motor-cortex context.\n",
        encoding="utf-8",
    )
