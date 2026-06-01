"""
EEG preprocessing pipeline.

Steps applied to each subject:
  1. Re-reference to average reference
  2. Band-pass filter: 0.5-40 Hz  (removes DC drift and high-freq noise)
  3. Notch filter: 50 Hz or 60 Hz (powerline)
  4. Epoch into fixed-length windows (default 2s, no overlap for train; 0.5s stride for stream)
  5. Amplitude-based artifact rejection: drop epochs > threshold uV
  6. Z-score normalize each epoch per channel

The 2-second window at 256 Hz = 512 samples per channel.
With 19 EEG channels (10-20 system), each epoch is (19, 512).
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Optional

import mne
import numpy as np

mne.set_log_level("WARNING")

SFREQ_TARGET = 256  # resample target Hz
WINDOW_SEC = 2.0
OVERLAP_SEC = 0.0  # training: no overlap
STREAM_STRIDE_SEC = 0.5  # streaming simulation stride
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
NOTCH_FREQ = 50.0
AMPLITUDE_THRESH_UV = 150.0  # reject epochs exceeding this


def preprocess_raw(
    raw: mne.io.BaseRaw,
    notch_freq: float = NOTCH_FREQ,
    sfreq_target: int = SFREQ_TARGET,
) -> mne.io.BaseRaw:
    """Apply filtering and resampling in-place."""
    # pick EEG channels only
    raw.pick_types(eeg=True, verbose=False)

    # re-reference to average
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # band-pass
    raw.filter(
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        method="fir",
        fir_window="hamming",
        verbose=False,
    )

    # notch filter for powerline
    raw.notch_filter(freqs=notch_freq, verbose=False)

    # resample to target sfreq
    if raw.info["sfreq"] != sfreq_target:
        raw.resample(sfreq_target, verbose=False)

    return raw


def epoch_raw(
    raw: mne.io.BaseRaw,
    window_sec: float = WINDOW_SEC,
    stride_sec: float = OVERLAP_SEC,
    amplitude_thresh_uv: float = AMPLITUDE_THRESH_UV,
) -> np.ndarray:
    """
    Cut raw EEG into fixed-length windows.

    Returns:
      epochs: shape (n_epochs, n_channels, n_samples)
              n_samples = window_sec * sfreq_target
    """
    sfreq = raw.info["sfreq"]
    n_window = int(window_sec * sfreq)
    n_stride = int((stride_sec if stride_sec > 0 else window_sec) * sfreq)

    data = raw.get_data(units="uV")  # (n_channels, n_times)
    n_ch, n_times = data.shape

    starts = range(0, n_times - n_window + 1, n_stride)
    epochs = []
    for start in starts:
        window = data[:, start : start + n_window]
        # artifact rejection: drop if any channel exceeds threshold
        if np.max(np.abs(window)) > amplitude_thresh_uv:
            continue
        epochs.append(window)

    if not epochs:
        return np.empty((0, n_ch, n_window), dtype=np.float32)

    arr = np.stack(epochs, axis=0).astype(np.float32)  # (N, C, T)
    # per-epoch, per-channel z-score normalization
    mean = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1, keepdims=True) + 1e-8
    arr = (arr - mean) / std
    return arr


def process_subject(
    record: dict,
    window_sec: float = WINDOW_SEC,
    stride_sec: float = 0.0,
) -> Optional[dict]:
    """
    Full pipeline for one subject.

    Returns dict with:
      subject_id, label, group, epochs (N, C, T)
    """
    raw = record["raw"]
    try:
        raw = preprocess_raw(raw)
        epochs = epoch_raw(raw, window_sec=window_sec, stride_sec=stride_sec)
    except Exception as exc:
        print(f"  [error] sub-{record['subject_id']}: {exc}")
        return None

    if epochs.shape[0] == 0:
        print(f"  [skip] sub-{record['subject_id']}: 0 valid epochs after rejection")
        return None

    return {
        "subject_id": record["subject_id"],
        "label": record["label"],
        "group": record["group"],
        "epochs": epochs,
        "sfreq": SFREQ_TARGET,
        "window_sec": window_sec,
    }


def save_processed(out_dir: Path, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        path = out_dir / f"sub-{rec['subject_id']}.pkl"
        with open(path, "wb") as f:
            pickle.dump(rec, f)
    # write manifest
    manifest = [
        {
            "subject_id": r["subject_id"],
            "label": r["label"],
            "group": r["group"],
            "n_epochs": r["epochs"].shape[0],
            "shape": list(r["epochs"].shape),
        }
        for r in records
    ]
    import json
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(records)} subjects to {out_dir}")


def load_processed(data_dir: Path) -> list[dict]:
    records = []
    for pkl in sorted(data_dir.glob("sub-*.pkl")):
        with open(pkl, "rb") as f:
            records.append(pickle.load(f))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from data.loader import load_dataset

    print(f"Loading dataset from {args.data_dir}...")
    raw_records = load_dataset(args.data_dir, verbose=True)

    print(f"Preprocessing {len(raw_records)} subjects...")
    processed = []
    for rec in raw_records:
        result = process_subject(rec)
        if result is not None:
            processed.append(result)
            print(
                f"  sub-{result['subject_id']} ({result['group']}): "
                f"{result['epochs'].shape[0]} epochs, shape {result['epochs'].shape}"
            )

    save_processed(Path(args.out_dir), processed)
