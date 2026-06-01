"""
Parallel Power Spectral Density computation for EEG.

Implements the Welch periodogram across all channels simultaneously using
vectorized FFT operations. Two backends:

  CuPy (CUDA)  — GPU parallel: all channels FFT'd in one cuFFT kernel call
  NumPy (CPU)  — Vectorized: all channels via numpy.fft.rfft simultaneously,
                 faster than scipy.signal.welch per-channel loop

The scipy.signal.welch baseline processes 19 channels sequentially (~2ms/window).
The NumPy vectorized path reduces that by removing Python loop overhead (~0.3ms).
The CuPy path offloads the FFT to GPU, amortizing transfer cost at batch_size >= 8.

Usage:
    from features.cuda_psd import psd_batch, band_integrate, BACKEND
    psd = psd_batch(epoch, sfreq=256, nperseg=256)   # (n_ch, n_freqs)
    powers = band_integrate(psd, freqs, BANDS)         # (n_ch, n_bands)

BACKEND is set automatically: 'cupy' if available, else 'numpy'.
"""

import numpy as np

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

try:
    import cupy as cp
    BACKEND = "cupy"
except ImportError:
    cp = None
    BACKEND = "numpy"


def _hann_window(nperseg: int) -> np.ndarray:
    return np.hanning(nperseg).astype(np.float32)


def _psd_numpy(
    epoch: np.ndarray, sfreq: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Welch PSD for all channels using numpy FFT.

    Args:
      epoch: (n_channels, n_samples) float32
      sfreq: sampling rate Hz
      nperseg: samples per FFT segment

    Returns:
      freqs: (n_freqs,)
      psd: (n_channels, n_freqs) — one-sided PSD, power/Hz units
    """
    n_ch, n_samples = epoch.shape
    hop = nperseg // 2
    window = _hann_window(nperseg)
    win_power = np.sum(window ** 2)

    n_segments = max(1, (n_samples - nperseg) // hop + 1)
    # Build segment matrix: (n_ch, n_segments, nperseg)
    segments = np.stack(
        [epoch[:, k * hop : k * hop + nperseg] for k in range(n_segments)],
        axis=1,
    )  # (n_ch, n_segs, nperseg)

    # Apply Hann window
    segments = segments * window[np.newaxis, np.newaxis, :]

    # FFT across last axis
    fft_out = np.fft.rfft(segments, axis=-1)  # (n_ch, n_segs, n_freqs)
    power = (np.abs(fft_out) ** 2) / (sfreq * win_power)

    # Double-sided → one-sided (exclude DC and Nyquist from doubling)
    n_freqs = fft_out.shape[-1]
    psd = power.mean(axis=1)  # (n_ch, n_freqs)
    psd[:, 1:-1] *= 2.0

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq).astype(np.float32)
    return freqs, psd.astype(np.float32)


def _psd_cupy(
    epoch: np.ndarray, sfreq: float, nperseg: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    CUDA-accelerated Welch PSD using CuPy (cuFFT backend).

    All channels are FFT'd in a single cuFFT call — no Python loop.
    Requires: pip install cupy-cudaXXX (match your CUDA version)
    """
    epoch_gpu = cp.asarray(epoch, dtype=cp.float32)
    n_ch, n_samples = epoch_gpu.shape
    hop = nperseg // 2
    window_gpu = cp.hanning(nperseg).astype(cp.float32)
    win_power = float(cp.sum(window_gpu ** 2))

    n_segments = max(1, (n_samples - nperseg) // hop + 1)
    segments = cp.stack(
        [epoch_gpu[:, k * hop : k * hop + nperseg] for k in range(n_segments)],
        axis=1,
    )
    segments = segments * window_gpu[cp.newaxis, cp.newaxis, :]

    fft_out = cp.fft.rfft(segments, axis=-1)  # single cuFFT call
    power = (cp.abs(fft_out) ** 2) / (sfreq * win_power)
    psd = power.mean(axis=1)
    psd[:, 1:-1] *= 2.0

    freqs = cp.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    return cp.asnumpy(freqs).astype(np.float32), cp.asnumpy(psd).astype(np.float32)


def psd_batch(
    epoch: np.ndarray, sfreq: float = 256.0, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD for all channels simultaneously.

    Args:
      epoch: (n_channels, n_samples) float32
      sfreq: sampling frequency Hz
      nperseg: FFT segment length (default: sfreq, i.e., 1-second segments)

    Returns:
      freqs: (n_freqs,)
      psd: (n_channels, n_freqs)
    """
    if nperseg is None:
        nperseg = min(epoch.shape[1], int(sfreq))

    if BACKEND == "cupy":
        return _psd_cupy(epoch, sfreq, nperseg)
    return _psd_numpy(epoch, sfreq, nperseg)


def band_integrate(
    psd: np.ndarray, freqs: np.ndarray, bands: dict | None = None
) -> np.ndarray:
    """
    Integrate PSD over frequency bands.

    Args:
      psd: (n_channels, n_freqs)
      freqs: (n_freqs,)
      bands: dict of {name: (lo_hz, hi_hz)} — defaults to BANDS

    Returns:
      powers: (n_channels, n_bands) absolute band powers
    """
    if bands is None:
        bands = BANDS
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    result = []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            result.append(_trapz(psd[:, mask], freqs[mask], axis=-1))
        else:
            result.append(np.zeros(psd.shape[0], dtype=np.float32))
    return np.stack(result, axis=1).astype(np.float32)  # (n_ch, n_bands)


def band_power_epoch_fast(
    epoch: np.ndarray, sfreq: float = 256.0
) -> np.ndarray:
    """
    Drop-in replacement for features.extractor.band_power_epoch().
    Uses parallel FFT (numpy or cupy) instead of per-channel scipy.welch loop.

    Returns: (n_channels * n_bands,) float32 — relative band powers
    """
    freqs, psd = psd_batch(epoch, sfreq)
    total = psd.sum(axis=-1, keepdims=True) + 1e-10  # (n_ch, 1)
    abs_powers = band_integrate(psd, freqs)           # (n_ch, n_bands)
    rel_powers = (abs_powers / total).astype(np.float32)
    return rel_powers.ravel()
