"""
Full LOSO (Leave-One-Subject-Out) benchmark.

For each subject in the dataset:
  1. Load the LOSO fold model trained without that subject
  2. Run all 3 inference modes (stream/adaptive/adaptive_adaskip) on that subject's data
  3. Record per-fold: accuracy, F1, P50/P95/P99 latency, skip rate, deadline miss rate

This produces the rigorous per-subject breakdown needed for:
  - Reproducing Miltiadous et al. (2023) paper accuracy on ds004504
  - Showing per-subject variance (critical for clinical AI credibility)
  - Proving 0% deadline misses holds across all 88 subjects

Output:
  results/loso_88subject_benchmark.csv — 88 × n_modes rows
  results/loso_88subject_summary.json  — mean ± std across all folds
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import onnxruntime as ort

from features.extractor import extract_features
from infer.stream import StreamConfig, StreamingInferenceEngine
from preprocess.pipeline import load_processed

N_FEATURES = 95  # 19 channels × 5 bands
DEADLINE_MS = 100.0
N_WARMUP = 10
MODES = ["stream", "adaptive", "adaptive_adaskip"]


def _load_fold_model_as_ort(model_name: str, subject_id: str, ckpt_dir: Path):
    """Load fold pickle, convert to ONNX, return ORT session (in-memory)."""
    pkl_path = ckpt_dir / model_name / f"fold_{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"No fold model: {pkl_path}")
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    init_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onnx_bytes = convert_sklearn(
        model, initial_types=init_type, options={"zipmap": False}, target_opset=17
    ).SerializeToString()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(onnx_bytes, sess_options=opts, providers=["CPUExecutionProvider"])


def _bench_subject(
    session: ort.InferenceSession,
    record: dict,
    mode: str,
    config: StreamConfig,
) -> dict:
    """Run one subject through the streaming engine in one mode."""
    engine = StreamingInferenceEngine(session, config, feature_type="band_power")
    epochs = record["epochs"]  # (N, C, T)
    n_ep, n_ch, n_t = epochs.shape
    continuous = epochs.reshape(n_ch, n_ep * n_t)
    true_label = record["label"]

    window_results = engine.run_stream(continuous, mode=mode)

    latencies = [r.latency_ms for r in window_results if not r.skipped]
    preds = [r.prediction for r in window_results]
    skips = sum(1 for r in window_results if r.skipped)
    misses = sum(1 for r in window_results if r.deadline_missed)
    n_win = len(window_results)

    lats = np.array(latencies) if latencies else np.array([0.0])
    true_arr = np.full(n_win, true_label)
    pred_arr = np.array(preds)

    return {
        "p50_ms": float(np.percentile(lats, 50)),
        "p95_ms": float(np.percentile(lats, 95)),
        "p99_ms": float(np.percentile(lats, 99)),
        "mean_ms": float(lats.mean()),
        "throughput_wps": float(1000.0 / lats.mean()) if lats.mean() > 0 else 0.0,
        "accuracy": float(accuracy_score(true_arr, pred_arr)),
        "macro_f1": float(f1_score(true_arr, pred_arr, average="macro", zero_division=0)),
        "skip_rate_pct": float(skips / n_win * 100) if n_win > 0 else 0.0,
        "deadline_miss_pct": float(misses / n_win * 100) if n_win > 0 else 0.0,
        "n_windows": n_win,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["svm_rbf", "random_forest", "mlp_sklearn"])
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = StreamConfig(
        sfreq=256,
        window_sec=2.0,
        stride_sec=0.5,
        deadline_ms=DEADLINE_MS,
        adaptive_threshold=0.98,
        confidence_threshold=args.confidence_threshold,
        use_confidence_gate=True,
    )

    records = load_processed(Path(args.data_dir))
    print(f"Loaded {len(records)} subjects")

    all_rows = []
    summaries = {}

    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        ckpt_dir = Path(args.model_dir)
        fold_dir = ckpt_dir / model_name
        if not fold_dir.exists():
            print(f"  [skip] No checkpoint dir: {fold_dir}")
            continue

        model_rows = []
        for rec in records:
            sub_id = rec["subject_id"]
            group = rec["group"]

            try:
                session = _load_fold_model_as_ort(model_name, sub_id, ckpt_dir)
            except FileNotFoundError:
                print(f"  [skip] sub-{sub_id}: no fold model")
                continue

            for mode in MODES:
                result = _bench_subject(session, rec, mode, config)
                row = {
                    "model": model_name,
                    "subject_id": sub_id,
                    "group": group,
                    "mode": mode,
                    **result,
                }
                model_rows.append(row)
                all_rows.append(row)
                print(
                    f"  sub-{sub_id} ({group}) | {mode:20s} | "
                    f"acc={result['accuracy']:.3f}  p50={result['p50_ms']:.3f}ms  "
                    f"skip={result['skip_rate_pct']:.1f}%"
                )

        # Per-model summary across all subjects
        for mode in MODES:
            mode_rows = [r for r in model_rows if r["mode"] == mode]
            if not mode_rows:
                continue
            accs = [r["accuracy"] for r in mode_rows]
            f1s = [r["macro_f1"] for r in mode_rows]
            p50s = [r["p50_ms"] for r in mode_rows]
            key = f"{model_name}_{mode}"
            summaries[key] = {
                "model": model_name,
                "mode": mode,
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "mean_f1": float(np.mean(f1s)),
                "mean_p50_ms": float(np.mean(p50s)),
                "n_subjects": len(mode_rows),
            }
            print(
                f"  [{model_name}/{mode}] acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
                f"f1={np.mean(f1s):.3f}  p50={np.mean(p50s):.3f}ms"
            )

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "loso_88subject_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    with open(out_dir / "loso_88subject_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n=== LOSO Summary ===")
    for key, s in summaries.items():
        print(f"  {key:40s}: acc={s['mean_acc']:.3f}±{s['std_acc']:.3f}  f1={s['mean_f1']:.3f}")
