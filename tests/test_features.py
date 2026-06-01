"""Tests for feature extraction."""

import numpy as np
import pytest

from features.extractor import (
    BANDS,
    band_power_epoch,
    extract_band_power,
    extract_features,
)

N_CH = 19
N_SAMPLES = 512
SFREQ = 256.0


class TestBandPowerEpoch:
    def test_output_length(self):
        epoch = np.random.randn(N_CH, N_SAMPLES).astype(np.float32)
        feats = band_power_epoch(epoch, SFREQ)
        assert feats.shape == (N_CH * len(BANDS),)

    def test_relative_power_all_positive(self):
        # band power features must be non-negative (they are power ratios)
        epoch = np.random.randn(N_CH, N_SAMPLES).astype(np.float32)
        feats = band_power_epoch(epoch, SFREQ)
        assert np.all(feats >= 0.0), "Band power features must be non-negative"

    def test_relative_power_each_between_0_and_1(self):
        # each relative power value must be in [0, 1] since it's band/total
        epoch = np.random.randn(N_CH, N_SAMPLES).astype(np.float32)
        feats = band_power_epoch(epoch, SFREQ)
        assert np.all(feats <= 1.0 + 1e-6), "Relative band power must be <= 1"

    def test_dominant_band_detected(self):
        # signal dominated by 10Hz (alpha band: 8-13Hz)
        t = np.linspace(0, 2.0, N_SAMPLES)
        signal = (np.sin(2 * np.pi * 10 * t) * 50).astype(np.float32)
        epoch = np.stack([signal] * N_CH)
        feats = band_power_epoch(epoch, SFREQ)
        # alpha is band index 2 (delta=0, theta=1, alpha=2, beta=3, gamma=4)
        alpha_idx = 2
        for ch in range(N_CH):
            ch_feats = feats[ch * len(BANDS) : (ch + 1) * len(BANDS)]
            assert ch_feats[alpha_idx] == max(ch_feats), f"Alpha not dominant in ch {ch}"

    def test_batch_extraction_shape(self):
        epochs = np.random.randn(10, N_CH, N_SAMPLES).astype(np.float32)
        X = extract_band_power(epochs, SFREQ)
        assert X.shape == (10, N_CH * len(BANDS))


class TestExtractFeatures:
    def _make_records(self, n_subjects: int = 5) -> list[dict]:
        records = []
        for i in range(n_subjects):
            epochs = np.random.randn(4, N_CH, N_SAMPLES).astype(np.float32)
            records.append(
                {"subject_id": str(i), "label": i % 3, "group": "C", "epochs": epochs, "sfreq": SFREQ}
            )
        return records

    def test_band_power_shapes(self):
        records = self._make_records(5)
        X, y, groups = extract_features(records, "band_power")
        assert X.shape == (20, N_CH * len(BANDS))
        assert y.shape == (20,)
        assert groups.shape == (20,)

    def test_raw_shapes(self):
        records = self._make_records(5)
        X, y, groups = extract_features(records, "raw")
        assert X.shape == (20, N_CH, N_SAMPLES)

    def test_group_alignment(self):
        records = self._make_records(3)
        X, y, groups = extract_features(records, "band_power")
        # all epochs from subject 0 should have label records[0]['label']
        mask = groups == "0"
        assert np.all(y[mask] == records[0]["label"])

    def test_unknown_feature_type_raises(self):
        records = self._make_records(2)
        with pytest.raises(ValueError, match="Unknown feature type"):
            extract_features(records, "unknown_type")
