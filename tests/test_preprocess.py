"""Tests for EEG preprocessing pipeline."""

import numpy as np
import pytest

from preprocess.pipeline import (
    AMPLITUDE_THRESH_UV,
    SFREQ_TARGET,
    WINDOW_SEC,
    epoch_raw,
)


def make_fake_raw(n_channels: int = 19, duration_sec: float = 30.0, sfreq: float = 256.0):
    """Create a mock MNE Raw object with sine wave data."""
    import mne

    n_times = int(duration_sec * sfreq)
    t = np.linspace(0, duration_sec, n_times)
    data = np.stack([np.sin(2 * np.pi * 10 * t) * 20 for _ in range(n_channels)])  # 20uV sine
    info = mne.create_info(ch_names=[f"CH{i}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # MNE expects volts
    return raw


class TestEpochRaw:
    def test_output_shape(self):
        raw = make_fake_raw(n_channels=5, duration_sec=20.0, sfreq=256.0)
        epochs = epoch_raw(raw, window_sec=2.0, stride_sec=0.0)
        # 20s / 2s = 10 epochs (no overlap)
        assert epochs.ndim == 3
        assert epochs.shape[1] == 5   # channels
        assert epochs.shape[2] == 512  # 2s * 256Hz

    def test_amplitude_rejection(self):
        import mne

        n_ch, sfreq, dur = 5, 256, 10
        n_times = dur * sfreq
        # half windows are huge amplitude, half are normal
        data = np.zeros((n_ch, n_times))
        # first 5s: normal 20uV
        data[:, : n_times // 2] = 20e-6
        # second 5s: artifact 500uV
        data[:, n_times // 2 :] = 500e-6
        info = mne.create_info([f"C{i}" for i in range(n_ch)], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)

        epochs = epoch_raw(raw, window_sec=2.0, stride_sec=0.0, amplitude_thresh_uv=150.0)
        # only windows from the first 5s should survive
        # first 5s => 2 full 2s windows
        assert epochs.shape[0] <= 3, f"Expected <=3 epochs, got {epochs.shape[0]}"

    def test_no_epochs_when_all_artifacts(self):
        import mne

        n_ch, sfreq, dur = 5, 256, 10
        data = np.ones((n_ch, dur * sfreq)) * 500e-6  # all artifact
        info = mne.create_info([f"C{i}" for i in range(n_ch)], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        epochs = epoch_raw(raw, window_sec=2.0, amplitude_thresh_uv=150.0)
        assert epochs.shape[0] == 0

    def test_normalization(self):
        raw = make_fake_raw(n_channels=3, duration_sec=10.0, sfreq=256.0)
        epochs = epoch_raw(raw, window_sec=2.0)
        # each epoch-channel should be ~zero mean after z-score
        means = epochs.mean(axis=-1)  # (N, C)
        assert np.allclose(means, 0.0, atol=1e-5)

    def test_overlap_produces_more_epochs(self):
        raw = make_fake_raw(n_channels=3, duration_sec=20.0, sfreq=256.0)
        no_overlap = epoch_raw(raw, window_sec=2.0, stride_sec=0.0)
        with_overlap = epoch_raw(raw, window_sec=2.0, stride_sec=0.5)
        assert with_overlap.shape[0] > no_overlap.shape[0]
