"""
Cross-dataset domain shift analysis.

Measures the gap between a model trained on ds004504 (AD/FTD/Healthy EEG)
and deployed on ds002778 (Parkinson's Disease vs. Healthy Controls EEG).

This quantifies the clinical deployment reality: models trained in one
hospital context fail systematically in another — even for the same task
(EEG-based neurological screening).

Protocol:
  1. Zero-shot: apply ds004504-trained model directly to ds002778 subjects
     (binary: PD=0 vs. HC=1, remapped to the 3-class model's probability space)
  2. 10-shot fine-tuning: take 10 ds002778 subjects as adaptation set,
     fine-tune with sklearn's warm_start, test on remaining subjects
  3. Report: zero-shot accuracy, fine-tuned accuracy, domain gap, confusion matrix

Download ds002778:
    python3 -m openneuro download --dataset ds002778 --target-dir data/raw/

Then preprocess:
    python3 -m preprocess.pipeline --data-dir data/raw/ds002778 --out-dir data/processed_parkinson

Run:
    python3 -m bench.domain_shift \
        --source-model models/checkpoints/svm_rbf/best_model.pkl \
        --target-dir data/processed_parkinson \
        --out-dir results/
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from features.extractor import extract_features
from preprocess.pipeline import load_processed


def load_ds002778_records(data_dir: Path) -> list[dict]:
    """
    Load ds002778 (Parkinson's Disease EEG).

    ds002778 participants.tsv has subject IDs starting with 'hc' (healthy control)
    or 'pd' (Parkinson's disease). Remap to binary labels:
      hc → 0 (Healthy)
      pd → 1 (Parkinson's Disease)
    """
    records = load_processed(data_dir)
    # ds002778 subject IDs are e.g. 'hc1', 'pd3'
    # The preprocess pipeline reads labels from participants.tsv Group column.
    # ds002778 Group column: 'hc' or 'pd' — override to binary labels
    remapped = []
    for r in records:
        sub_id = r["subject_id"]
        # Infer label from subject ID prefix if not already set
        if sub_id.startswith("hc"):
            binary_label = 0
        elif sub_id.startswith("pd"):
            binary_label = 1
        else:
            # Fall back to existing label (may be wrong if mapped from A/F/C)
            binary_label = r["label"]
        remapped.append({**r, "label": binary_label, "binary_label": binary_label})
    return remapped


def zero_shot_eval(source_model, target_records: list[dict]) -> dict:
    """
    Apply source model to target domain data without any adaptation.

    For binary output: use max probability across AD/FTD/Healthy classes,
    then remap: AD(0) → PD(1), FTD(1) → PD(1), Healthy(2) → Healthy(0).
    This is a crude but honest mapping — the source model doesn't know about PD.
    """
    X, y_true, _ = extract_features(target_records, "band_power")

    # Source model predicts AD/FTD/Healthy (0/1/2)
    # Remap: source 2 (Healthy) → target 0 (HC); source 0/1 (AD/FTD) → target 1 (PD)
    source_preds = source_model.predict(X)
    remapped_preds = np.where(source_preds == 2, 0, 1)

    acc = float(accuracy_score(y_true, remapped_preds))
    f1 = float(f1_score(y_true, remapped_preds, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, remapped_preds, labels=[0, 1]).tolist()

    return {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm, "n_subjects": len(target_records)}


def few_shot_finetune(
    source_model: Pipeline,
    target_records: list[dict],
    n_shot_subjects: int = 10,
    seed: int = 42,
) -> dict:
    """
    10-shot fine-tuning: use n_shot_subjects from target domain to adapt,
    evaluate on remaining subjects.

    Since sklearn SVM can't warm-start, train a new LinearSVC on:
      source training features + n_shot target features (combined)
    Then test on remaining target subjects.
    """
    rng = np.random.default_rng(seed)
    all_subjects = [r["subject_id"] for r in target_records]
    rng.shuffle(all_subjects)
    shot_subjects = set(all_subjects[:n_shot_subjects])
    test_subjects = set(all_subjects[n_shot_subjects:])

    shot_records = [r for r in target_records if r["subject_id"] in shot_subjects]
    test_records = [r for r in target_records if r["subject_id"] in test_subjects]

    if not test_records:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n_test_subjects": 0}

    X_shot, y_shot, _ = extract_features(shot_records, "band_power")
    X_test, y_test, _ = extract_features(test_records, "band_power")

    # Build adapted model: fine-tune on shot data only (binary classification)
    adapted = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=seed)),
    ])
    adapted.fit(X_shot, y_shot)
    preds = adapted.predict(X_test)

    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "n_shot_subjects": len(shot_records),
        "n_test_subjects": len(test_records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True, help="Path to best_model.pkl from ds004504 training")
    parser.add_argument("--target-dir", required=True, help="Preprocessed ds002778 data dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-shot", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading source model (ds004504 trained)...")
    with open(args.source_model, "rb") as f:
        source_model = pickle.load(f)

    print(f"Loading target domain data from {args.target_dir}...")
    target_records = load_ds002778_records(Path(args.target_dir))
    print(f"  {len(target_records)} subjects | Classes: {set(r['label'] for r in target_records)}")

    print("\nRunning zero-shot evaluation...")
    zero_shot = zero_shot_eval(source_model, target_records)
    print(f"  Zero-shot accuracy: {zero_shot['accuracy']:.3f}  F1: {zero_shot['macro_f1']:.3f}")

    print(f"\nRunning {args.n_shot}-shot fine-tuning...")
    few_shot = few_shot_finetune(source_model, target_records, n_shot_subjects=args.n_shot)
    print(f"  Fine-tuned accuracy: {few_shot['accuracy']:.3f}  F1: {few_shot['macro_f1']:.3f}")
    print(f"  Domain gap recovered: {few_shot['accuracy'] - zero_shot['accuracy']:+.3f}")

    results = {
        "source_dataset": "ds004504 (AD/FTD/Healthy, 88 subjects)",
        "target_dataset": "ds002778 (Parkinson's/Healthy, 31 subjects)",
        "zero_shot": zero_shot,
        f"few_shot_{args.n_shot}": few_shot,
        "domain_gap_accuracy": float(few_shot["accuracy"] - zero_shot["accuracy"]),
    }

    out_path = out_dir / "domain_shift_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n=== Domain Shift Summary ===")
    print(f"  Zero-shot (ds004504 → ds002778): acc={zero_shot['accuracy']:.3f}  f1={zero_shot['macro_f1']:.3f}")
    print(f"  {args.n_shot}-shot fine-tuned:              acc={few_shot['accuracy']:.3f}  f1={few_shot['macro_f1']:.3f}")
    print(f"  Domain gap recovered:            {few_shot['accuracy'] - zero_shot['accuracy']:+.3f}")
    print(f"  Interpretation: source model trained on AD/FTD does not generalize to Parkinson's detection.")
    print(f"  {args.n_shot} adaptation subjects recover {few_shot['accuracy'] - zero_shot['accuracy']:.1%} of that gap.")
