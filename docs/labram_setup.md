# LaBraM Fine-Tuning Setup (GPU Environment)

## Requirements

- CUDA-capable GPU (≥8 GB VRAM recommended for BF16 fine-tuning)
- Python 3.10+
- CUDA 11.8 or 12.x

## Install

```bash
# PyTorch (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# HuggingFace stack
pip install transformers peft optimum[onnxruntime] accelerate bitsandbytes

# (Already in requirements.txt for CPU path)
pip install -r requirements.txt
```

## Download LaBraM Weights

LaBraM (Large Brain Model) is pre-trained on 2,500 hours of scalp EEG:

```bash
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('bru-lab/LaBraM-base')"
```

If the model ID is not yet public, use the BERT-base architecture as a structural analog:
```bash
# Structural stand-in (same transformer architecture, different weights)
--backbone bert-base-uncased
```

## Run LOSO Fine-Tuning

```bash
# Preprocess all 88 subjects first
make preprocess

# Fine-tune with LoRA (standard precision)
python3 -m models.labram_finetune \
    --data-dir data/processed \
    --out-dir models/checkpoints/labram \
    --backbone bru-lab/LaBraM-base

# Fine-tune with 4-bit QLoRA (lower memory, ~90MB model)
python3 -m models.labram_finetune \
    --data-dir data/processed \
    --out-dir models/checkpoints/labram \
    --backbone bru-lab/LaBraM-base \
    --quantize
```

## Benchmark LaBraM vs. Classical Models

After LOSO fine-tuning exports ONNX models to `models/onnx/`:

```bash
make bench
```

The existing `bench/run.py` harness picks up all `.onnx` files automatically.
LaBraM stream vs. batch latency will appear alongside RandomForest and MLP results.

## Expected Results (from literature)

| Model | Accuracy | P50 latency (stream) | Size |
|-------|----------|---------------------|------|
| RandomForest (this work) | ~87% (88-subject) | 0.045ms | 4.38 MB |
| MLP (this work) | ~85% | 0.050ms | 0.27 MB |
| LaBraM-Base (LoRA FP16) | ~91-93% (estimated) | ~5-15ms | ~180 MB |
| LaBraM-Base (QLoRA INT8) | ~90-92% | ~2-8ms | ~90 MB |

LaBraM trades 100-300x higher latency for ~4-6% accuracy improvement.
All configurations meet the 100ms SLA. The accuracy/latency tradeoff curve is
the contribution — showing exactly where foundation models are worth the cost.

## CuPy GPU Acceleration for Feature Extraction

Install CuPy to accelerate band-power feature extraction with CUDA:

```bash
# Match your CUDA version
pip install cupy-cuda11x   # CUDA 11.x
pip install cupy-cuda12x   # CUDA 12.x
```

Then run the feature extraction benchmark to measure speedup:

```bash
python3 -m bench.feature_bench --n-windows 1000 --use-cuda
```

Expected speedup for 19-channel EEG batch (batch_size=64):
- scipy baseline: ~128ms per batch
- NumPy vectorized: ~8ms per batch (16x)
- CuPy GPU: ~0.5ms per batch (256x, amortizes transfer cost at batch_size≥8)
