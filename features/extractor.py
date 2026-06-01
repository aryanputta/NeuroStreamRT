"""
Feature extraction for EEG epochs.

Three feature sets:
  1. band_power  — relative band power per channel (delta, theta, alpha, beta, gamma)
  2. csp         — Common Spatial Pattern log-variance features (pairwise binary, then concat)
  3. raw_epochs  — pass-through for end-to-end DL models

Band definitions (Hz):
  delta:  0.5-4
  theta:  4-8
  alpha:  8-13
  beta:   13-30
  gamma:  30-40
"""

import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def band_power_epoch(
    epoch: np.ndarray, sfreq: float = 256.0
) -> np.ndarray:
    """
    Compute relative band power for one epoch.

    Args:
      epoch: (n_channels, n_samples) float32
      sfreq: sampling frequency

    Returns:
      features: (n_channels * n_bands,) float32
    """
    n_ch, n_samples = epoch.shape
    n_per_seg = min(n_samples, int(sfreq * 1.0))  # 1s segments

    features = []
    for ch in range(n_ch):
        freqs, psd = welch(epoch[ch], fs=sfreq, nperseg=n_per_seg)
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        total_power = _trapz(psd, freqs) + 1e-10
        ch_features = []
        for lo, hi in BANDS.values():
            mask = (freqs >= lo) & (freqs < hi)
            band_pwr = _trapz(psd[mask], freqs[mask])
            ch_features.append(band_pwr / total_power)
        features.extend(ch_features)

    return np.array(features, dtype=np.float32)


def extract_band_power(
    epochs: np.ndarray, sfreq: float = 256.0
) -> np.ndarray:
    """
    Extract band power for all epochs.

    Args:
      epochs: (N, n_channels, n_samples)

    Returns:
      X: (N, n_channels * n_bands)
    """
    return np.stack([band_power_epoch(ep, sfreq) for ep in epochs], axis=0)


def extract_features(
    records: list[dict], feature_type: str = "band_power"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X, y, groups arrays from preprocessed records.

    Args:
      records: list of processed subject dicts
      feature_type: 'band_power' or 'raw'

    Returns:
      X: (total_epochs, features)
      y: (total_epochs,)  int labels
      groups: (total_epochs,)  subject_id per epoch (for group CV)
    """
    X_parts, y_parts, g_parts = [], [], []
    for rec in records:
        epochs = rec["epochs"]  # (N, C, T)
        label = rec["label"]
        sub_id = rec["subject_id"]
        sfreq = rec.get("sfreq", 256.0)

        if feature_type == "band_power":
            feats = extract_band_power(epochs, sfreq)
        elif feature_type == "raw":
            feats = epochs  # (N, C, T) — for DL
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        n = len(feats)
        X_parts.append(feats)
        y_parts.append(np.full(n, label, dtype=np.int64))
        g_parts.append(np.full(n, sub_id))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(g_parts, axis=0)
    return X, y, groups
