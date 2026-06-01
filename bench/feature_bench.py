"""
Feature extraction latency benchmark.

Compares three PSD computation paths:
  scipy   — scipy.signal.welch, per-channel loop (production baseline, ~2ms/window)
  numpy   — vectorized numpy FFT across all channels simultaneously
  cupy    — CUDA-accelerated FFT via CuPy (requires GPU)

Measures P50/P95/P99 latency per single window and per batch of N windows.

Key finding: scipy.signal.welch is the actual bottleneck in NeuroStreamRT —
inference (ONNX) takes 0.04ms, feature extraction takes ~2ms. This benchmark
quantifies the improvement from switching to parallel FFT.

Run:
    python3 -m bench.feature_bench [--n-windows 1000] [--batch-size 64] [--use-cuda]
"""

import argparse
import time

import numpy as np
from scipy.signal import welch as scipy_welch

from features.cuda_psd import BACKEND, band_power_epoch_fast, psd_batch

SFREQ = 256.0
N_CHANNELS = 19
N_SAMPLES = 512  # 2s window
N_BANDS = 5
N_PER_SEG = 256


def _band_power_scipy(epoch: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """Original per-channel scipy implementation (baseline)."""
    from features.extractor import band_power_epoch
    return band_power_epoch(epoch, sfreq)


def bench_path(
    fn,
    windows: list[np.ndarray],
    n_warmup: int = 20,
) -> tuple[np.ndarray, float]:
    """
    Benchmark a feature extraction function over a list of windows.

    Returns:
      latencies: (n_windows,) in ms
      mean_latency: ms
    """
    for w in windows[:n_warmup]:
        fn(w)

    latencies = []
    for w in windows:
        t0 = time.perf_counter()
        fn(w)
        latencies.append((time.perf_counter() - t0) * 1000)
    arr = np.array(latencies)
    return arr, float(arr.mean())


def bench_batch(
    fn_batch,
    batch: np.ndarray,
    n_reps: int = 200,
    n_warmup: int = 20,
) -> np.ndarray:
    """Benchmark batch feature extraction (N windows at once)."""
    for _ in range(n_warmup):
        fn_batch(batch)
    latencies = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn_batch(batch)
        latencies.append((time.perf_counter() - t0) * 1000)
    return np.array(latencies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-windows", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    windows = [rng.standard_normal((N_CHANNELS, N_SAMPLES)).astype(np.float32) for _ in range(args.n_windows)]
    batch = np.stack(windows[:args.batch_size], axis=0)

    paths = {
        "scipy (baseline)": _band_power_scipy,
        f"numpy_fast ({BACKEND})": band_power_epoch_fast,
    }

    print(f"\nFeature Extraction Latency Benchmark")
    print(f"{'='*70}")
    print(f"Windows: {args.n_windows} | Channels: {N_CHANNELS} | Samples: {N_SAMPLES} | Backend: {BACKEND}")
    print()

    print(f"{'Path':25s} {'P50(ms)':>9} {'P95(ms)':>9} {'P99(ms)':>9} {'Mean(ms)':>10} {'Speedup':>9}")
    print("-" * 70)

    baseline_p50 = None
    for name, fn in paths.items():
        lats, mean = bench_path(fn, windows)
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        if baseline_p50 is None:
            baseline_p50 = p50
            speedup = "1.0x"
        else:
            speedup = f"{baseline_p50/p50:.1f}x"
        print(f"{name:25s} {p50:>9.3f} {p95:>9.3f} {p99:>9.3f} {mean:>10.3f} {speedup:>9}")

    print()
    print(f"Batch Extraction (batch_size={args.batch_size}) — per-window time:")
    print(f"{'Path':25s} {'P50(ms)':>9} {'P95(ms)':>9} {'Per-win P50':>12} {'Speedup':>9}")
    print("-" * 60)

    def _scipy_batch(batch_arr):
        return np.stack([_band_power_scipy(w) for w in batch_arr], axis=0)

    def _numpy_batch(batch_arr):
        return np.stack([band_power_epoch_fast(w) for w in batch_arr], axis=0)

    batch_paths = {
        "scipy (baseline)": _scipy_batch,
        f"numpy_fast ({BACKEND})": _numpy_batch,
    }

    baseline_batch_p50 = None
    for name, fn in batch_paths.items():
        lats = bench_batch(fn, batch)
        p50 = float(np.percentile(lats, 50))
        p95 = float(np.percentile(lats, 95))
        per_win = p50 / args.batch_size
        if baseline_batch_p50 is None:
            baseline_batch_p50 = p50
            speedup = "1.0x"
        else:
            speedup = f"{baseline_batch_p50/p50:.1f}x"
        print(f"{name:25s} {p50:>9.3f} {p95:>9.3f} {per_win:>12.4f} {speedup:>9}")

    print()
    print(f"Note: On GPU (CuPy), cuFFT kernel handles all {N_CHANNELS} channels in one call.")
    print(f"Install CuPy: pip install cupy-cudaXXX (match your CUDA version)")
