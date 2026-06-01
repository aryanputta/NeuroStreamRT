"""
Streaming EEG inference simulator.

Simulates real-time deployment: EEG arrives as a continuous stream at 256 Hz.
A sliding window of 2s (512 samples) advances by stride_samples each tick.
At each tick the inference pipeline runs and must complete within deadline_ms.

Modes:
  batch     — accumulate all windows, run inference once (offline baseline)
  stream    — process each window independently as it arrives
  adaptive  — skip inference on windows similar to previous (cosine similarity threshold)
              simulates early-exit / caching to reduce redundant compute

This module is used by bench/run.py to drive latency measurements.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import onnxruntime as ort

from features.extractor import band_power_epoch


@dataclass
class StreamConfig:
    sfreq: int = 256
    window_sec: float = 2.0
    stride_sec: float = 0.5
    deadline_ms: float = 100.0
    adaptive_threshold: float = 0.98


@dataclass
class WindowResult:
    window_idx: int
    latency_ms: float
    prediction: int
    confidence: float
    skipped: bool = False
    deadline_missed: bool = False


def build_ort_session(onnx_path: str, providers: Optional[list[str]] = None) -> ort.InferenceSession:
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1  # single-threaded for fair per-window timing
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)


class StreamingInferenceEngine:
    """Runs inference on an EEG stream window by window."""

    def __init__(
        self,
        session: ort.InferenceSession,
        config: StreamConfig,
        feature_type: str = "raw",
    ):
        self.session = session
        self.config = config
        self.feature_type = feature_type
        self._last_features: Optional[np.ndarray] = None
        self._input_name = session.get_inputs()[0].name

    def _extract(self, window: np.ndarray) -> np.ndarray:
        """Extract features from window (C, T) → model input."""
        if self.feature_type == "band_power":
            feats = band_power_epoch(window, self.config.sfreq)
            return feats[np.newaxis, :]  # (1, F)
        else:
            return window[np.newaxis, :, :]  # (1, C, T)

    def _infer(self, model_input: np.ndarray) -> tuple[int, float]:
        """Run ONNX inference, return (predicted_class, confidence)."""
        outputs = self.session.run(None, {self._input_name: model_input.astype(np.float32)})
        logits = outputs[0][0]  # (n_classes,)
        probs = _softmax(logits)
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        return pred, conf

    def _is_similar(self, features: np.ndarray) -> bool:
        """Return True if features are nearly identical to last window (adaptive skip)."""
        if self._last_features is None:
            return False
        sim = _cosine_sim(features.ravel(), self._last_features.ravel())
        return sim >= self.config.adaptive_threshold

    def run_stream(
        self,
        eeg_data: np.ndarray,
        mode: str = "stream",
    ) -> list[WindowResult]:
        """
        Process EEG data in streaming fashion.

        Args:
          eeg_data: (n_channels, n_times) continuous EEG
          mode: 'batch' | 'stream' | 'adaptive'

        Returns:
          list of WindowResult per window
        """
        n_window = int(self.config.window_sec * self.config.sfreq)
        n_stride = int(self.config.stride_sec * self.config.sfreq)
        n_ch, n_times = eeg_data.shape

        starts = list(range(0, n_times - n_window + 1, n_stride))
        windows = [eeg_data[:, s : s + n_window] for s in starts]

        if mode == "batch":
            return self._run_batch(windows)
        elif mode == "stream":
            return self._run_per_window(windows, adaptive=False)
        elif mode == "adaptive":
            return self._run_per_window(windows, adaptive=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _run_batch(self, windows: list[np.ndarray]) -> list[WindowResult]:
        """Offline batch: stack all windows, one inference call."""
        t0 = time.perf_counter()
        all_inputs = np.stack([self._extract(w)[0] for w in windows], axis=0)
        outputs = self.session.run(None, {self._input_name: all_inputs.astype(np.float32)})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_window_ms = elapsed_ms / len(windows)

        logits_batch = outputs[0]
        results = []
        for i, logits in enumerate(logits_batch):
            probs = _softmax(logits)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            results.append(
                WindowResult(
                    window_idx=i,
                    latency_ms=per_window_ms,
                    prediction=pred,
                    confidence=conf,
                    deadline_missed=per_window_ms > self.config.deadline_ms,
                )
            )
        return results

    def _run_per_window(
        self, windows: list[np.ndarray], adaptive: bool
    ) -> list[WindowResult]:
        """Process each window independently."""
        results = []
        last_pred, last_conf = 0, 0.0

        for i, window in enumerate(windows):
            features = self._extract(window)

            if adaptive and self._is_similar(features):
                results.append(
                    WindowResult(
                        window_idx=i,
                        latency_ms=0.0,
                        prediction=last_pred,
                        confidence=last_conf,
                        skipped=True,
                    )
                )
                continue

            t0 = time.perf_counter()
            pred, conf = self._infer(features)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            self._last_features = features
            last_pred, last_conf = pred, conf

            results.append(
                WindowResult(
                    window_idx=i,
                    latency_ms=latency_ms,
                    prediction=pred,
                    confidence=conf,
                    deadline_missed=latency_ms > self.config.deadline_ms,
                )
            )
        return results


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)
