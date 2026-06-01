"""
EEG loader for OpenNeuro ds004504.

ds004504 structure:
  sub-<N>/eeg/sub-<N>_task-eyesclosed_eeg.set  (EEGLAB .set files)
  participants.tsv: Group column = A (Alzheimer), F (FTD), C (Control)

Labels:
  A -> 0  (Alzheimer's Disease)
  F -> 1  (Frontotemporal Dementia)
  C -> 2  (Healthy Control)
"""

import json
import os
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd

LABEL_MAP = {"A": 0, "F": 1, "C": 2}
CLASS_NAMES = ["Alzheimers", "FTD", "Healthy"]


def load_participants(data_dir: Path) -> pd.DataFrame:
    tsv = data_dir / "participants.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found in {data_dir}")
    df = pd.read_csv(tsv, sep="\t")
    # column is 'Group' with values A, F, C
    df["label"] = df["Group"].map(LABEL_MAP)
    return df


def load_raw_eeg(
    data_dir: Path,
    subject_id: str,
    task: str = "eyesclosed",
    preload: bool = True,
    verbose: bool = False,
) -> Optional[mne.io.BaseRaw]:
    """Load raw EEG for one subject. Returns None if file not found."""
    eeg_dir = data_dir / f"sub-{subject_id}" / "eeg"
    candidates = list(eeg_dir.glob(f"*task-{task}*_eeg.set"))
    if not candidates:
        if verbose:
            print(f"  [skip] sub-{subject_id}: no .set file found")
        return None
    raw = mne.io.read_raw_eeglab(str(candidates[0]), preload=preload, verbose=verbose)
    return raw


def load_dataset(
    data_dir: str | Path,
    subjects: Optional[list[str]] = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Load all subjects. Returns list of dicts with keys:
      subject_id, label, group, raw
    """
    data_dir = Path(data_dir)
    participants = load_participants(data_dir)

    records = []
    for _, row in participants.iterrows():
        sub_id = str(row["participant_id"]).replace("sub-", "")
        if subjects is not None and sub_id not in subjects:
            continue
        label = row["label"]
        group = row["Group"]
        raw = load_raw_eeg(data_dir, sub_id, verbose=verbose)
        if raw is None:
            continue
        records.append(
            {"subject_id": sub_id, "label": label, "group": group, "raw": raw}
        )

    if verbose:
        print(f"Loaded {len(records)} subjects from {data_dir}")
    return records
