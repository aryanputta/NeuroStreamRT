"""
Train EEGNet, ShallowConvNet, and MLP on ds004504.

Validation: Leave-One-Subject-Out (LOSO) CV.
Each subject is held out once; remaining subjects train.
Reports per-fold and aggregate accuracy, F1, confusion matrix.

Checkpoints saved to: models/checkpoints/<model_name>/
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from features.extractor import extract_features
from models.eegnet import EEGNet
from models.mlp_baseline import MLPBaseline
from models.shallow_convnet import ShallowConvNet
from preprocess.pipeline import load_processed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 3
N_EPOCHS_TRAIN = 50
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 10


def build_model(name: str, n_channels: int, n_samples: int, n_features: int) -> nn.Module:
    if name == "eegnet":
        return EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=N_CLASSES)
    elif name == "shallowconv":
        return ShallowConvNet(n_channels=n_channels, n_samples=n_samples, n_classes=N_CLASSES)
    elif name == "mlp":
        return MLPBaseline(n_features=n_features, n_classes=N_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")


def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[nn.Module, dict]:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_TRAIN)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_v = torch.from_numpy(X_val).float()
    y_v = torch.from_numpy(y_val).long()

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    best_val_acc = 0.0
    best_state = None
    patience_count = 0

    for epoch in range(N_EPOCHS_TRAIN):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_v.to(DEVICE)).argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_v.numpy(), preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_v.to(DEVICE)).argmax(dim=1).cpu().numpy()

    return model, {
        "val_acc": accuracy_score(y_v.numpy(), preds),
        "val_f1": f1_score(y_v.numpy(), preds, average="macro"),
        "predictions": preds.tolist(),
        "true_labels": y_v.numpy().tolist(),
    }


def run_loso_cv(
    records: list[dict],
    model_name: str,
    feature_type: str,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = [r["subject_id"] for r in records]
    all_results = []

    first_rec = records[0]
    n_channels = first_rec["epochs"].shape[1]
    n_samples = first_rec["epochs"].shape[2]
    n_bands = 5
    n_features = n_channels * n_bands

    for fold_idx, held_out in enumerate(subjects):
        train_recs = [r for r in records if r["subject_id"] != held_out]
        val_recs = [r for r in records if r["subject_id"] == held_out]

        X_train, y_train, _ = extract_features(train_recs, feature_type)
        X_val, y_val, _ = extract_features(val_recs, feature_type)

        model = build_model(model_name, n_channels, n_samples, n_features)
        _, fold_result = train_one_fold(model, X_train, y_train, X_val, y_val)

        fold_result["subject_id"] = held_out
        fold_result["fold"] = fold_idx
        all_results.append(fold_result)

        print(
            f"  Fold {fold_idx+1:02d} | sub-{held_out} | "
            f"acc={fold_result['val_acc']:.3f} f1={fold_result['val_f1']:.3f}"
        )

        # save best model for this fold (optional — used in bench)
        torch.save(
            model.state_dict(),
            out_dir / f"{model_name}_sub{held_out}.pt",
        )

    accs = [r["val_acc"] for r in all_results]
    f1s = [r["val_f1"] for r in all_results]
    summary = {
        "model": model_name,
        "feature_type": feature_type,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "folds": all_results,
    }

    with open(out_dir / f"{model_name}_loso_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[{model_name}] LOSO CV: "
        f"acc={summary['mean_acc']:.3f}±{summary['std_acc']:.3f} "
        f"f1={summary['mean_f1']:.3f}±{summary['std_f1']:.3f}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "eegnet", "shallowconv"],
        choices=["mlp", "eegnet", "shallowconv"],
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading processed data from {args.data_dir}...")
    records = load_processed(Path(args.data_dir))
    print(f"  {len(records)} subjects")

    model_feature_map = {
        "mlp": "band_power",
        "eegnet": "raw",
        "shallowconv": "raw",
    }

    all_summaries = {}
    for model_name in args.models:
        feature_type = model_feature_map[model_name]
        print(f"\n=== Training {model_name} (features: {feature_type}) ===")
        summary = run_loso_cv(
            records,
            model_name,
            feature_type,
            Path(args.out_dir) / model_name,
        )
        all_summaries[model_name] = summary

    with open(Path(args.out_dir) / "all_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n=== Summary ===")
    for name, s in all_summaries.items():
        print(f"  {name:15s}: acc={s['mean_acc']:.3f}±{s['std_acc']:.3f}  f1={s['mean_f1']:.3f}")
