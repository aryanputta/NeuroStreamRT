"""
Export PyTorch models to ONNX and apply static INT8 quantization.

Quantization levels:
  fp32  — baseline full precision
  int8  — post-training static quantization via ONNX Runtime
  int4  — simulated via dynamic quantization (no hardware INT4 kernel needed)

For each model + quantization combo, exports:
  models/onnx/<model_name>_fp32.onnx
  models/onnx/<model_name>_int8.onnx
"""

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader

from features.extractor import extract_features
from models.eegnet import EEGNet
from models.mlp_baseline import MLPBaseline
from models.shallow_convnet import ShallowConvNet
from preprocess.pipeline import load_processed

N_CHANNELS = 19
N_SAMPLES = 512
N_CLASSES = 3
N_BANDS = 5
N_FEATURES_MLP = N_CHANNELS * N_BANDS


def load_model(model_name: str, checkpoint_path: Path) -> torch.nn.Module:
    if model_name == "eegnet":
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
    elif model_name == "shallowconv":
        model = ShallowConvNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
    elif model_name == "mlp":
        model = MLPBaseline(n_features=N_FEATURES_MLP, n_classes=N_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def export_onnx(
    model: torch.nn.Module,
    model_name: str,
    out_path: Path,
    dummy_input: torch.Tensor,
) -> None:
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        verbose=False,
    )
    onnx.checker.check_model(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Exported {out_path.name} ({size_mb:.2f} MB)")


class EEGCalibrationReader(CalibrationDataReader):
    """Feeds calibration batches to ONNX Runtime quantizer."""

    def __init__(self, data: np.ndarray, batch_size: int = 32):
        self.data = data.astype(np.float32)
        self.batch_size = batch_size
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.data):
            return None
        batch = self.data[self._idx : self._idx + self.batch_size]
        self._idx += self.batch_size
        return {"input": batch}

    def rewind(self):
        self._idx = 0


def quantize_to_int8(
    fp32_path: Path,
    int8_path: Path,
    calibration_data: np.ndarray,
) -> None:
    reader = EEGCalibrationReader(calibration_data, batch_size=32)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False,
    )
    size_mb = int8_path.stat().st_size / 1e6
    print(f"  Quantized {int8_path.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory with .pt checkpoints")
    parser.add_argument("--data-dir", required=True, help="Processed data dir for calibration")
    parser.add_argument("--out-dir", required=True, help="Output ONNX directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_processed(Path(args.data_dir))
    # use first 20% as calibration data
    calib_recs = records[: max(1, len(records) // 5)]

    model_configs = {
        "eegnet": ("raw", torch.zeros(1, N_CHANNELS, N_SAMPLES)),
        "shallowconv": ("raw", torch.zeros(1, N_CHANNELS, N_SAMPLES)),
        "mlp": ("band_power", torch.zeros(1, N_FEATURES_MLP)),
    }

    for model_name, (feat_type, dummy) in model_configs.items():
        ckpt_dir = Path(args.model_dir) / model_name
        checkpoints = list(ckpt_dir.glob(f"{model_name}_sub*.pt"))
        if not checkpoints:
            print(f"[skip] No checkpoints found for {model_name}")
            continue

        # export first checkpoint as representative model
        # in production you'd average weights or pick best fold
        ckpt = sorted(checkpoints)[0]
        print(f"\n=== {model_name} (from {ckpt.name}) ===")

        model = load_model(model_name, ckpt)
        fp32_path = out_dir / f"{model_name}_fp32.onnx"
        int8_path = out_dir / f"{model_name}_int8.onnx"

        export_onnx(model, model_name, fp32_path, dummy)

        # calibration data
        calib_X, _, _ = extract_features(calib_recs, feat_type)
        quantize_to_int8(fp32_path, int8_path, calib_X)
