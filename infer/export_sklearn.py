"""
Export sklearn pipelines to ONNX for deployment benchmarking.

Uses skl2onnx (sklearn-onnx) to convert fitted sklearn pipelines.
Produces:
  models/onnx/<model_name>_fp32.onnx  — full precision (float32)

ONNX INT8 quantization via onnxruntime.quantization is applied post-export:
  models/onnx/<model_name>_int8.onnx  — INT8 quantized
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_sklearn_to_onnx(
    model,
    n_features: int,
    output_path: Path,
) -> None:
    """Convert fitted sklearn pipeline to ONNX."""
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={"zipmap": False},
        target_opset=17,
    )
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Exported {output_path.name} ({size_mb:.3f} MB)")


def quantize_to_int8_dynamic(fp32_path: Path, int8_path: Path) -> None:
    """Apply dynamic INT8 quantization — no calibration data needed for sklearn models."""
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    size_mb = int8_path.stat().st_size / 1e6
    print(f"  Quantized {int8_path.name} ({size_mb:.3f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Dir with best_model.pkl files")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-features", type=int, default=95, help="19 channels * 5 bands")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    for subdir in sorted(model_dir.iterdir()):
        if not subdir.is_dir():
            continue
        pkl_path = subdir / "best_model.pkl"
        if not pkl_path.exists():
            print(f"[skip] {subdir.name}: no best_model.pkl")
            continue

        model_name = subdir.name
        print(f"\n=== {model_name} ===")

        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        fp32_path = out_dir / f"{model_name}_fp32.onnx"
        int8_path = out_dir / f"{model_name}_int8.onnx"

        try:
            export_sklearn_to_onnx(model, args.n_features, fp32_path)
            quantize_to_int8_dynamic(fp32_path, int8_path)
        except Exception as e:
            print(f"  [error] {e}")
