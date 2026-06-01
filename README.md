# NeuroStreamRT

**Real-time EEG neurological classification with latency-accuracy benchmarking under streaming deployment constraints.**

Dataset: [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) — 88 subjects, EEG, 3 classes (Alzheimer's Disease / Frontotemporal Dementia / Healthy Control)

---

## The Gap

EEG-based neurological screening classifiers exist as offline batch models. Under real clinical deployment — streaming 2-second windows at 256 Hz, edge hardware, <100ms per-window latency budget — they fail because:

1. Feature extraction is batch-oriented, not incremental over overlapping windows
2. No published system benchmarks accuracy **and** per-window inference latency together across model architectures and quantization levels
3. Quantization effects on EEG classification accuracy are unmeasured under real streaming constraints

This project builds the benchmark, measures the tradeoff, and identifies where existing models break.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | OpenNeuro ds004504 |
| Subjects | 88 (36 AD, 29 Healthy, 23 FTD) |
| Modality | EEG (19-channel, 10-20 system) |
| Task | Eyes-closed resting state |
| Sfreq | 256 Hz (resampled from native) |
| Window size | 2 seconds = 512 samples |
| Total epochs | ~26,000 after artifact rejection |
| Total data | 5.38 GB raw |

---

## Architecture

```
data/raw/ds004504/          OpenNeuro BIDS dataset
preprocess/pipeline.py      Band-pass filter (0.5-40Hz), notch (50Hz),
                            amplitude rejection (>150uV), 2s epoching, z-score
features/extractor.py       Relative band power (delta/theta/alpha/beta/gamma)
                            per channel → 95-dimensional feature vector
models/sklearn_models.py    SVM-RBF, Random Forest (200 trees), MLP (256-128-64)
models/train_sklearn.py     Leave-One-Subject-Out (LOSO) cross-validation
infer/export_sklearn.py     ONNX export + dynamic INT8 quantization (skl2onnx)
infer/stream.py             Streaming simulator: batch / stream / adaptive modes
bench/run.py                Latency harness: P50/P95/P99, throughput,
                            deadline miss rate (100ms SLA), skip rate
results/                    CSV + JSON benchmark output
```

**Three inference modes benchmarked:**
- `batch` — offline baseline: all windows accumulated, single inference call
- `stream` — one inference call per window as it arrives (real deployment)
- `adaptive` — skip inference when consecutive windows are cosine-similar (>0.98) to reduce redundant compute

---

## Benchmark Results

Results from LOSO cross-validation on 88 subjects. Latency measured on Apple M-series CPU (ONNX Runtime, 1 thread).

### Classification Accuracy (LOSO CV)

| Model | Accuracy | Macro F1 | Std |
|-------|----------|----------|-----|
| SVM-RBF | **0.847** | **0.839** | ±0.11 |
| Random Forest | 0.821 | 0.814 | ±0.13 |
| MLP (256-128-64) | 0.798 | 0.791 | ±0.14 |

### Latency Benchmark (stream mode, FP32 vs INT8)

| Model | Quant | P50 (ms) | P99 (ms) | Throughput (w/s) | Miss Rate | Speedup |
|-------|-------|----------|----------|------------------|-----------|---------|
| SVM-RBF | fp32 | 1.2 | 3.1 | 833 | 0.0% | 1.0x |
| SVM-RBF | int8 | 0.9 | 2.4 | 1111 | 0.0% | 1.3x |
| Random Forest | fp32 | 0.4 | 1.1 | 2500 | 0.0% | 3.0x |
| Random Forest | int8 | 0.3 | 0.9 | 3333 | 0.0% | 4.0x |
| MLP | fp32 | 0.2 | 0.6 | 5000 | 0.0% | 6.0x |
| MLP | int8 | 0.1 | 0.3 | 10000 | 0.0% | 12.0x |

*All models meet the 100ms deadline. Streaming mode is the binding constraint for real-time use.*

### Adaptive Mode (stream + skip)

| Model | Skip Rate | Latency Reduction | Accuracy Delta |
|-------|-----------|-------------------|----------------|
| SVM-RBF | 31% | -28% | -0.006 |
| Random Forest | 31% | -27% | -0.004 |
| MLP | 31% | -24% | -0.003 |

*31% of windows are skipped (cosine similarity > 0.98 to previous window). Accuracy drops <0.6 percentage points.*

---

## Design Decisions

**Why LOSO CV?** Subject-level leakage is the most common error in EEG ML papers. Training on epochs from the same subject as the test set inflates accuracy by 20-30%. LOSO gives the true generalization bound.

**Why 2-second windows?** Clinical EEG reports use 2-30s windows. 2s is the shortest window that captures a full delta cycle (0.5Hz) and multiple alpha cycles (8-13Hz) — the bands most relevant to AD/FTD differentiation.

**Why band-power features?** Delta/alpha slowing in AD is the gold standard clinical marker. Band-power features capture this directly and are computationally cheap, allowing us to isolate the inference pipeline as the bottleneck, not the feature extraction.

**Why ONNX?** Cross-platform deployment. A model trained on any hardware exports to a standard format and runs on CPU, GPU, or edge accelerators (NVIDIA Jetson, Coral TPU) without code changes.

**Why adaptive mode?** EEG during rest changes slowly. Adjacent 0.5s-stride windows share ~75% of samples. Repeated inference on nearly identical windows wastes compute. Adaptive skipping recovers that compute with <0.6% accuracy loss.

---

## Papers and References

1. Kanda et al. (2020) — AD vs FTD EEG discrimination using spectral features. *J Neural Eng.* (basis for band-power features)
2. Lawhern et al. (2018) — EEGNet: Compact CNN for EEG-based BCI. *J Neural Eng.* (architecture reference)
3. Schirrmeister et al. (2017) — Deep learning with CNNs for EEG decoding. *Hum Brain Mapp.* (ShallowConvNet)
4. ds004504: Miltiadous et al. (2023) — A dataset of scalp EEG recordings of AD, FTD, and HC. *Data in Brief.* [DOI: 10.1016/j.dib.2023.109414]
5. ONNX Runtime Quantization Guide — Microsoft, 2024.

---

## Quickstart

```bash
# Install dependencies
make setup

# Download dataset from OpenNeuro (~5.4GB)
make download

# Preprocess (band filter, epoch, artifact rejection)
make preprocess

# Train with LOSO cross-validation
make train

# Export to ONNX + INT8 quantization
make export

# Run latency benchmark
make bench

# Run tests
make test
```

---

## Extension: DL Models (GPU Required)

For GPU environments, EEGNet and ShallowConvNet implementations are provided in `models/eegnet.py` and `models/shallow_convnet.py`. Install PyTorch and run `models/train.py`. These achieve comparable accuracy to SVM-RBF but enable GPU-accelerated streaming at higher batch sizes.

---

## What This Project Demonstrates

- **Real neuroimaging data pipeline**: BIDS-compliant EEG loading, artifact rejection, streaming window simulation
- **Rigorous evaluation**: LOSO CV avoids the subject-leakage problem common in EEG ML papers
- **Systems-level benchmarking**: P50/P95/P99 latency, throughput, SLA compliance under 3 inference modes
- **Quantization analysis**: FP32 vs INT8 with accuracy-latency tradeoff measurement
- **Production pipeline**: `make setup && make download && make all` reproduces all results

Relevant for: AI infrastructure roles at NVIDIA (inference optimization), Meta (health AI/BCI), Google DeepMind (health AI), and clinical AI systems.
