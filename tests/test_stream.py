"""Tests for streaming inference engine and latency benchmark logic."""

import io
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from infer.stream import (
    StreamConfig,
    StreamingInferenceEngine,
    _cosine_sim,
    _softmax,
)


class TestSoftmax:
    def test_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_max_is_argmax(self):
        logits = np.array([0.1, 5.0, 0.5])
        probs = _softmax(logits)
        assert np.argmax(probs) == 1

    def test_numerical_stability(self):
        logits = np.array([1000.0, 1001.0, 999.0])
        probs = _softmax(logits)
        assert np.isfinite(probs).all()


class TestCosineSim:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_sim(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_zero_vector(self):
        a = np.zeros(5)
        b = np.ones(5)
        result = _cosine_sim(a, b)
        assert np.isfinite(result)


def make_mock_session(n_classes: int = 3):
    """Create a mock ONNX session that returns random logits."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]

    def fake_run(output_names, input_dict):
        batch_size = list(input_dict.values())[0].shape[0]
        logits = np.random.randn(batch_size, n_classes).astype(np.float32)
        return [logits]

    session.run = fake_run
    return session


class TestStreamingInferenceEngine:
    def setup_method(self):
        self.config = StreamConfig(sfreq=256, window_sec=2.0, stride_sec=0.5, deadline_ms=100.0)
        self.session = make_mock_session()
        self.engine = StreamingInferenceEngine(self.session, self.config, feature_type="raw")

    def _make_eeg(self, duration_sec: float = 10.0) -> np.ndarray:
        n_ch = 19
        n_times = int(duration_sec * self.config.sfreq)
        return np.random.randn(n_ch, n_times).astype(np.float32)

    def test_stream_mode_produces_results(self):
        eeg = self._make_eeg(10.0)
        results = self.engine.run_stream(eeg, mode="stream")
        assert len(results) > 0

    def test_batch_mode_produces_results(self):
        eeg = self._make_eeg(10.0)
        results = self.engine.run_stream(eeg, mode="batch")
        assert len(results) > 0

    def test_adaptive_skips_some_windows(self):
        # Use very low threshold so almost all windows are skipped after first
        config = StreamConfig(sfreq=256, window_sec=2.0, stride_sec=0.5, adaptive_threshold=0.0)
        engine = StreamingInferenceEngine(self.session, config, feature_type="raw")
        # Make constant EEG — adjacent windows will be nearly identical
        eeg = np.ones((19, 2560), dtype=np.float32)
        results = engine.run_stream(eeg, mode="adaptive")
        skipped = sum(1 for r in results if r.skipped)
        # With threshold=0.0 everything is "similar" after first window
        assert skipped > 0

    def test_window_indices_are_sequential(self):
        eeg = self._make_eeg(8.0)
        results = self.engine.run_stream(eeg, mode="stream")
        indices = [r.window_idx for r in results]
        assert indices == list(range(len(results)))

    def test_stream_window_count_matches_expectation(self):
        eeg = self._make_eeg(10.0)  # 10s
        results = self.engine.run_stream(eeg, mode="stream")
        # 10s with 2s windows, 0.5s stride -> (10 - 2) / 0.5 + 1 = 17
        assert 15 <= len(results) <= 18

    def test_latency_recorded_per_window(self):
        eeg = self._make_eeg(6.0)
        results = self.engine.run_stream(eeg, mode="stream")
        for r in results:
            assert r.latency_ms >= 0.0

    def test_predictions_in_valid_range(self):
        eeg = self._make_eeg(6.0)
        results = self.engine.run_stream(eeg, mode="stream")
        for r in results:
            assert 0 <= r.prediction <= 2

    def test_confidence_between_0_and_1(self):
        eeg = self._make_eeg(6.0)
        results = self.engine.run_stream(eeg, mode="stream")
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_unknown_mode_raises(self):
        eeg = self._make_eeg(4.0)
        with pytest.raises(ValueError, match="Unknown mode"):
            self.engine.run_stream(eeg, mode="invalid_mode")

    def test_adaptive_adaskip_mode_runs(self):
        """adaptive_adaskip mode should produce results without error."""
        eeg = self._make_eeg(8.0)
        results = self.engine.run_stream(eeg, mode="adaptive_adaskip")
        assert len(results) > 0

    def test_confidence_gate_prevents_skip_on_low_confidence(self):
        """Window must NOT be skipped when last confidence is below threshold."""
        config = StreamConfig(
            sfreq=256, window_sec=2.0, stride_sec=0.5,
            adaptive_threshold=0.0,      # always "similar" (0.0 = always trigger)
            confidence_threshold=0.99,   # threshold very high
            use_confidence_gate=True,
        )
        engine = StreamingInferenceEngine(self.session, config, feature_type="raw")
        # Constant EEG — cosine sim will be 1.0 (similar)
        eeg = np.ones((19, 2560), dtype=np.float32)
        results = engine.run_stream(eeg, mode="adaptive_adaskip")
        # First window always runs inference (no prior), subsequent windows:
        # similarity triggers but confidence gate holds them if conf < 0.99
        # Since mock session returns random logits, softmax confidence rarely > 0.99
        # so very few (or zero) windows should be skipped
        skipped = [r for r in results if r.skipped]
        total = len(results)
        skip_rate = len(skipped) / total if total > 0 else 0.0
        # With threshold=0.99 and random logits, expect very low skip rate (<20%)
        assert skip_rate < 0.20, f"Expected low skip rate with high threshold, got {skip_rate:.2f}"

    def test_confidence_gate_disabled_skips_like_plain_adaptive(self):
        """With gate disabled, adaptive_adaskip should skip identically to adaptive."""
        config = StreamConfig(
            sfreq=256, window_sec=2.0, stride_sec=0.5,
            adaptive_threshold=0.0,  # always "similar"
            use_confidence_gate=False,
        )
        engine_adaskip = StreamingInferenceEngine(self.session, config, feature_type="raw")
        engine_adaptive = StreamingInferenceEngine(self.session, config, feature_type="raw")
        eeg = np.ones((19, 2048), dtype=np.float32)
        res_adaskip = engine_adaskip.run_stream(eeg, mode="adaptive_adaskip")
        res_adaptive = engine_adaptive.run_stream(eeg, mode="adaptive")
        skips_adaskip = sum(1 for r in res_adaskip if r.skipped)
        skips_adaptive = sum(1 for r in res_adaptive if r.skipped)
        # Both should skip the same windows (gate disabled means same logic)
        assert skips_adaskip == skips_adaptive
