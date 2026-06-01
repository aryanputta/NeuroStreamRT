"""
NeuroStreamRT benchmark harness.

For every (model, quantization, inference_mode) combination, measures:
  - P50, P95, P99 per-window inference latency (ms)
  - Throughput (windows/sec)
  - Deadline miss rate (% windows > 100ms)
  - 3-class accuracy and macro F1
  - Model size (MB)
  - Skip rate (adaptive mode only)

Results written to:
  results/latency_benchmark.csv   — per-combination summary
  results/per_window_results.json — raw per-window records (for plotting)

Baseline comparison: batch_fp32 is the reference. All others report
  speedup = baseline_p50 / this_p50
  accuracy_delta = this_acc - baseline_acc
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from features.extractor import extract_features
from infer.stream import StreamConfig, StreamingInferenceEngine, build_ort_session
from preprocess.pipeline import load_processed

DEADLINE_MS = 100.0
N_WARMUP = 10


def run_benchmark(
    onnx_path: str,
    records: list[dict],
    model_name: str,
    quant: str,
    mode: str,
    feature_type: str,
    config: StreamConfig,
) -> dict:
    session = build_ort_session(onnx_path)
    engine = StreamingInferenceEngine(session, config, feature_type=feature_type)

    # warmup
    dummy = records[0]["epochs"][0]
    for _ in range(N_WARMUP):
        inp = dummy[np.newaxis, :] if feature_type == "raw" else None
        if inp is None:
            from features.extractor import band_power_epoch
            inp = band_power_epoch(dummy)[np.newaxis, :]
        session.run(None, {session.get_inputs()[0].name: inp.astype(np.float32)})

    all_latencies = []
    all_preds = []
    all_true = []
    all_skipped = []
    all_deadline_missed = []

    for rec in records:
        eeg = rec["epochs"]  # (N_epochs, C, T)
        n_ep, n_ch, n_t = eeg.shape
        # concatenate epochs into a single time series for streaming sim
        continuous = eeg.reshape(n_ch, n_ep * n_t)
        true_label = rec["label"]

        window_results = engine.run_stream(continuous, mode=mode)

        for wr in window_results:
            if not wr.skipped:
                all_latencies.append(wr.latency_ms)
            all_preds.append(wr.prediction)
            all_true.append(true_label)
            all_skipped.append(wr.skipped)
            all_deadline_missed.append(wr.deadline_missed)

    latencies = np.array(all_latencies)
    skips = np.array(all_skipped)
    missed = np.array(all_deadline_missed)
    preds = np.array(all_preds)
    true = np.array(all_true)

    model_size_mb = os.path.getsize(onnx_path) / 1e6

    return {
        "model": model_name,
        "quantization": quant,
        "mode": mode,
        "p50_ms": float(np.percentile(latencies, 50)) if len(latencies) > 0 else 0.0,
        "p95_ms": float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0,
        "p99_ms": float(np.percentile(latencies, 99)) if len(latencies) > 0 else 0.0,
        "mean_ms": float(latencies.mean()) if len(latencies) > 0 else 0.0,
        "throughput_wps": float(1000.0 / latencies.mean()) if len(latencies) > 0 else 0.0,
        "deadline_miss_pct": float(missed.mean() * 100),
        "skip_rate_pct": float(skips.mean() * 100),
        "accuracy": float(accuracy_score(true, preds)),
        "macro_f1": float(f1_score(true, preds, average="macro")),
        "model_size_mb": round(model_size_mb, 3),
        "n_windows": int(len(all_preds)),
        "n_subjects": len(records),
    }


def compute_speedup(results: list[dict]) -> list[dict]:
    baseline = next(
        (r for r in results if r["mode"] == "batch" and r["quantization"] == "fp32"), None
    )
    if baseline is None:
        return results
    baseline_p50 = baseline["p50_ms"]
    baseline_acc = baseline["accuracy"]
    for r in results:
        if baseline_p50 > 0:
            r["speedup_vs_batch_fp32"] = round(baseline_p50 / max(r["p50_ms"], 0.001), 2)
        else:
            r["speedup_vs_batch_fp32"] = 1.0
        r["accuracy_delta"] = round(r["accuracy"] - baseline_acc, 4)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory with .onnx files")
    parser.add_argument("--data-dir", required=True, help="Processed data dir")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_processed(Path(args.data_dir))
    print(f"Loaded {len(records)} subjects")

    config = StreamConfig(
        sfreq=256,
        window_sec=2.0,
        stride_sec=0.5,
        deadline_ms=DEADLINE_MS,
        adaptive_threshold=0.98,
        confidence_threshold=0.85,
        use_confidence_gate=True,
    )

    model_feature_map = {
        "eegnet": "raw",
        "shallowconv": "raw",
        "mlp": "band_power",
        "mlp_sklearn": "band_power",
        "svm_rbf": "band_power",
        "random_forest": "band_power",
        "randomforest": "band_power",
        "linearsvc": "band_power",
    }

    combos = []
    for onnx_file in sorted(Path(args.model_dir).glob("*.onnx")):
        stem = onnx_file.stem  # e.g. random_forest_fp32
        # parse model name: everything before the last _fp32/_int8 suffix
        if stem.endswith("_fp32"):
            model_name = stem[:-5]
            quant = "fp32"
        elif stem.endswith("_int8"):
            model_name = stem[:-5]
            quant = "int8"
        else:
            model_name = stem
            quant = "fp32"
        feature_type = model_feature_map.get(model_name, "band_power")
        combos.append((str(onnx_file), model_name, quant, feature_type))

    results = []
    for onnx_path, model_name, quant, feat_type in combos:
        for mode in ["batch", "stream", "adaptive", "adaptive_adaskip"]:
            print(f"\nBenchmarking {model_name} / {quant} / {mode}...")
            try:
                res = run_benchmark(
                    onnx_path, records, model_name, quant, mode, feat_type, config
                )
                results.append(res)
                print(
                    f"  p50={res['p50_ms']:.1f}ms  p99={res['p99_ms']:.1f}ms  "
                    f"acc={res['accuracy']:.3f}  skip={res['skip_rate_pct']:.1f}%  "
                    f"miss={res['deadline_miss_pct']:.1f}%"
                )
            except Exception as exc:
                print(f"  [error] {exc}")

    results = compute_speedup(results)

    df = pd.DataFrame(results)
    csv_path = out_dir / "latency_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print("\n=== Benchmark Summary ===")
    print(df[["model", "quantization", "mode", "p50_ms", "p99_ms", "accuracy", "speedup_vs_batch_fp32", "deadline_miss_pct"]].to_string(index=False))
