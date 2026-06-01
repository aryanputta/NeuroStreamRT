# NeuroStreamRT

Real-time EEG neurological classification with latency-accuracy benchmarking under streaming deployment constraints.

Dataset: [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) — 88 subjects, EEG, 3 classes (Alzheimer's Disease / Frontotemporal Dementia / Healthy Control)

---

## Motivation

EEG-based neurological screening classifiers are evaluated offline. Under real deployment constraints — streaming 2-second windows at 256 Hz, edge hardware, <100ms per-window latency budget — the evaluation protocol changes. Three gaps exist:

1. Feature extraction is batch-oriented, not incremental over overlapping windows
2. No published benchmark measures accuracy and per-window inference latency jointly across model architectures
3. Quantization effects on EEG classification have not been measured under streaming constraints

This project builds that benchmark.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | OpenNeuro ds004504 |
| Subjects | 88 (36 AD, 29 Healthy, 23 FTD) |
| Modality | EEG, 19-channel 10-20 system |
| Task | Eyes-closed resting state |
| Sampling rate | 256 Hz (resampled from native) |
| Window | 2 seconds = 512 samples |
| Total epochs | ~26,000 after artifact rejection |
| Raw size | 5.38 GB |

---

## Structure

```
data/            BIDS loader for ds004504 (load_dataset, load_participants)
preprocess/      Band-pass 0.5-40Hz, notch 50Hz, amplitude rejection >150µV,
                 2s epoching, per-epoch per-channel z-score normalization
features/        Band-power extractor (delta/theta/alpha/beta/gamma) → 95 features
                 cuda_psd.py: vectorized batch FFT (NumPy + CuPy/CUDA)
                 Miltiadous feature set: delta/alpha ratio + inter-channel coherence
models/          SVM-RBF, RandomForest (200 trees), MLP (256-128-64) via sklearn
                 EEGNet, ShallowConvNet (PyTorch, GPU required)
                 LaBraM LoRA fine-tuning stub (GPU required, docs/labram_setup.md)
infer/           ONNX export, streaming simulator (batch/stream/adaptive/adaptive_adaskip)
bench/           Latency harness (P50/P95/P99, throughput, deadline miss rate)
                 loso_bench.py: per-fold LOSO latency + accuracy
                 feature_bench.py: feature extraction latency comparison
                 domain_shift.py: cross-dataset zero-shot + few-shot eval
results/         CSV/JSON benchmark output
```

**Inference modes:**
- `batch` — offline baseline: all windows at once
- `stream` — one inference call per window as it arrives
- `adaptive` — skip windows with cosine similarity > 0.98 to previous
- `adaptive_adaskip` — AdaSkip-style: skip only when similar AND last prediction confidence >= threshold (default 0.85)

---

## Results

All measurements on real data. Hardware: Apple M-series CPU, ONNX Runtime 1.23.2, 1 thread.

### Dataset (10-subject pilot)

| Metric | Value |
|--------|-------|
| Subjects | 10 |
| Epochs | 3,220 |
| Class distribution | AD=1,173  FTD=818  HC=1,229 |
| Features | 95 (19 channels × 5 bands) |
| Epochs/subject | 153–429 |

### Inference Latency (ONNX, 2,000 runs, 100 warmup)

| Model | Mode | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (w/s) | SLA Miss | Size (MB) |
|-------|------|----------|----------|----------|-----------------|----------|-----------|
| RandomForest (200t) | stream | 0.045 | 0.050 | 0.064 | 21,600 | 0.0% | 4.38 |
| RandomForest (200t) | batch/64 | 0.015 | 0.016 | 0.017 | 67,498 | 0.0% | 4.38 |
| MLP (256-128-64) | stream | 0.050 | 0.053 | 0.067 | 19,644 | 0.0% | 0.27 |
| MLP (256-128-64) | batch/64 | 0.004 | 0.004 | 0.004 | 281,122 | 0.0% | 0.27 |

SLA = 100ms. SLA miss rate = 0.0% across all configurations.

### Feature Extraction Latency (`features/cuda_psd.py`)

Profiling revealed that `scipy.signal.welch` (sequential per-channel) was the actual bottleneck — not model inference. Replaced with vectorized batch FFT across all 19 channels simultaneously.

| Implementation | Mode | P50 (ms) | Speedup |
|----------------|------|----------|---------|
| scipy.signal.welch | single window | 29.7 | 1.0x |
| numpy batch FFT | single window | 1.03 | 28.8x |
| scipy.signal.welch | batch/64 | 20.9 ms/win | 1.0x |
| numpy batch FFT | batch/64 | 0.50 ms/win | 41.8x |

CuPy CUDA path available for GPU acceleration (same interface).

### Batch vs. Stream Speedup (ONNX)

| Model | Stream P50 | Batch/64 P50 | Speedup |
|-------|-----------|--------------|---------|
| RandomForest | 0.045 ms | 0.015 ms | 3.0x |
| MLP | 0.050 ms | 0.004 ms | 12.5x |

---

## Design Notes

**LOSO CV** — Subject-level leakage inflates EEG accuracy by 20-30% in most published benchmarks. LOSO holds out one full subject per fold. Each fold is independent. This is the correct evaluation protocol for cross-subject generalization.

**2-second windows** — 2s captures a full delta cycle (0.5 Hz) and multiple alpha cycles (8-13 Hz), which are the primary biomarkers for AD/FTD. Shorter windows miss low-frequency structure; longer windows increase latency.

**Band-power features** — Delta/alpha power ratio is a validated clinical marker for AD. Computationally cheap (after the FFT optimization), which lets the benchmark isolate inference as the variable under study.

**ONNX** — Models export from sklearn via skl2onnx. ONNX Runtime runs on CPU, GPU, or edge accelerators without code changes. This separates training from deployment.

**AdaSkip confidence gate** — The original adaptive mode skips based on cosine similarity alone. When model confidence on the previous window was low, that cached prediction is not safe to reuse. The `adaptive_adaskip` mode adds a confidence threshold gate to prevent stale predictions from propagating.

---

## References

1. Miltiadous et al. (2023) — EEG dataset of AD, FTD, HC. *Data in Brief.* DOI: 10.1016/j.dib.2023.109414
2. Lawhern et al. (2018) — EEGNet: Compact CNN for EEG BCI. *J Neural Eng.*
3. Schirrmeister et al. (2017) — Deep learning for EEG decoding. *Hum Brain Mapp.*
4. Kanda et al. (2020) — AD vs FTD spectral feature discrimination. *J Neural Eng.*
5. Chen et al. (2024) — AdaSkip: Adaptive sublayer skipping for LLM inference acceleration.

---

## Quickstart

```bash
make setup           # install dependencies
make download        # fetch ds004504 from OpenNeuro (~5.4GB)
make preprocess      # filter, epoch, artifact reject → data/processed/
make train           # LOSO CV → models/checkpoints/
make export          # ONNX + INT8 → models/onnx/
make bench           # latency benchmark → results/latency_benchmark.csv
make loso-bench      # per-fold LOSO accuracy + latency
make feature-bench   # feature extraction latency comparison
make test            # 45 tests
```

**Cross-dataset domain shift (ds002778, Parkinson's EEG):**
```bash
make download-parkinson
make preprocess-parkinson
make domain-shift
```

**GPU / LaBraM fine-tuning:**
See `docs/labram_setup.md`.
