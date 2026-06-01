"""
LOSO cross-validation training for sklearn models.

Produces:
  models/checkpoints/<model_name>/fold_<sub_id>.pkl  — per-fold fitted model
  models/checkpoints/<model_name>/loso_results.json  — accuracy, F1, confusion matrix
  models/checkpoints/<model_name>/best_model.pkl     — model trained on all data
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from features.extractor import extract_features
from models.sklearn_models import MODEL_REGISTRY, get_model
from preprocess.pipeline import load_processed

N_CLASSES = 3


def run_loso_cv(
    records: list[dict],
    model_name: str,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = [r["subject_id"] for r in records]
    all_results = []

    for fold_idx, held_out in enumerate(subjects):
        train_recs = [r for r in records if r["subject_id"] != held_out]
        val_recs = [r for r in records if r["subject_id"] == held_out]

        X_train, y_train, _ = extract_features(train_recs, "band_power")
        X_val, y_val, _ = extract_features(val_recs, "band_power")

        model = get_model(model_name)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, preds)
        val_f1 = f1_score(y_val, preds, average="macro")

        # save fold model
        with open(out_dir / f"fold_{held_out}.pkl", "wb") as f:
            pickle.dump(model, f)

        fold_result = {
            "subject_id": held_out,
            "fold": fold_idx,
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "n_val_epochs": int(len(y_val)),
            "n_train_epochs": int(len(y_train)),
        }
        all_results.append(fold_result)

        print(
            f"  Fold {fold_idx+1:02d} | sub-{held_out} | "
            f"acc={val_acc:.3f}  f1={val_f1:.3f}  "
            f"(train={len(y_train)}, val={len(y_val)})"
        )

    # train final model on all data
    X_all, y_all, _ = extract_features(records, "band_power")
    final_model = get_model(model_name)
    final_model.fit(X_all, y_all)
    with open(out_dir / "best_model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    accs = [r["val_acc"] for r in all_results]
    f1s = [r["val_f1"] for r in all_results]

    # aggregate confusion matrix across all folds
    all_true, all_pred = [], []
    for i, fold_idx_sub in enumerate(subjects):
        val_recs = [r for r in records if r["subject_id"] == fold_idx_sub]
        X_val, y_val, _ = extract_features(val_recs, "band_power")
        fold_path = out_dir / f"fold_{fold_idx_sub}.pkl"
        with open(fold_path, "rb") as f:
            fold_model = pickle.load(f)
        preds = fold_model.predict(X_val)
        all_true.extend(y_val.tolist())
        all_pred.extend(preds.tolist())

    cm = confusion_matrix(all_true, all_pred, labels=list(range(N_CLASSES))).tolist()

    summary = {
        "model": model_name,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "min_acc": float(np.min(accs)),
        "max_acc": float(np.max(accs)),
        "confusion_matrix": cm,
        "class_labels": ["Alzheimers(0)", "FTD(1)", "Healthy(2)"],
        "n_subjects": len(subjects),
        "folds": all_results,
    }

    with open(out_dir / "loso_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[{model_name}] LOSO: "
        f"acc={summary['mean_acc']:.3f}±{summary['std_acc']:.3f}  "
        f"f1={summary['mean_f1']:.3f}±{summary['std_f1']:.3f}  "
        f"range=[{summary['min_acc']:.3f}, {summary['max_acc']:.3f}]"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
    )
    args = parser.parse_args()

    print(f"Loading processed data from {args.data_dir}...")
    records = load_processed(Path(args.data_dir))
    print(f"  {len(records)} subjects loaded")

    all_summaries = {}
    for model_name in args.models:
        print(f"\n=== Training {model_name} (LOSO CV) ===")
        summary = run_loso_cv(records, model_name, Path(args.out_dir) / model_name)
        all_summaries[model_name] = summary

    out_path = Path(args.out_dir) / "all_results.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n=== Summary ===")
    for name, s in all_summaries.items():
        print(f"  {name:20s}: acc={s['mean_acc']:.3f}±{s['std_acc']:.3f}  f1={s['mean_f1']:.3f}")
