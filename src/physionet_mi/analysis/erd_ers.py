"""Band-power and lateralization features from EEG trials."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from physionet_mi.constants import CLASS_TO_NAME

MU_BAND = (8, 13)
BETA_BAND = (13, 30)

CHANNEL_ALIASES = {
    "C3": ["C3"],
    "C4": ["C4"],
    "Cz": ["Cz"],
    "FC3": ["FC3"],
    "FC4": ["FC4"],
    "CP3": ["CP3"],
    "CP4": ["CP4"],
}


def _bandpower(channel_ts: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    f, pxx = welch(channel_ts, fs=sfreq, nperseg=min(256, len(channel_ts)))
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return 0.0
    return float(np.trapz(pxx[mask], f[mask]))


def compute_trial_bandpower(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    ch_names: list[str],
    sfreq: float,
) -> pd.DataFrame:
    """X: (N, C, T). Returns trial-level mu/beta power and lateralization index."""
    ch_index = {name: i for i, name in enumerate(ch_names)}
    rows = []
    for i in range(len(X)):
        trial = X[i]
        powers = {}
        for label, aliases in CHANNEL_ALIASES.items():
            idx = next((ch_index[a] for a in aliases if a in ch_index), None)
            if idx is None:
                powers[f"mu_{label}"] = np.nan
                powers[f"beta_{label}"] = np.nan
                continue
            ts = trial[idx]
            powers[f"mu_{label}"] = _bandpower(ts, sfreq, MU_BAND)
            powers[f"beta_{label}"] = _bandpower(ts, sfreq, BETA_BAND)

        c3_mu = powers.get("mu_C3", np.nan)
        c4_mu = powers.get("mu_C4", np.nan)
        li_mu = (c4_mu - c3_mu) / (c4_mu + c3_mu + 1e-12) if np.isfinite(c3_mu) and np.isfinite(c4_mu) else np.nan

        class_idx = int(y[i])
        label_name = CLASS_TO_NAME.get(class_idx, str(class_idx))
        if label_name == "left_hand":
            contra, ipsi = c4_mu, c3_mu
        else:
            contra, ipsi = c3_mu, c4_mu

        rows.append({
            "subject_id": int(groups[i]),
            "label": label_name,
            "mu_power_C3": c3_mu,
            "mu_power_C4": c4_mu,
            "beta_power_C3": powers.get("beta_C3", np.nan),
            "beta_power_C4": powers.get("beta_C4", np.nan),
            "contralateral_mu": contra,
            "ipsilateral_mu": ipsi,
            "lateralization_index_mu": li_mu,
        })
    return pd.DataFrame(rows)
