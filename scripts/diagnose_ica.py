#!/usr/bin/env python3
"""Diagnose ICA on a single subject's filtered checkpoint."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg.cli import resolve_datasets_arg, subject_num_from_id
from eeg.config import load_experiment
from eeg.io import load_checkpoint
from eeg.paths import resolve_checkpoint_path
from eeg.preprocessing import stage_ica


def parse_args():
    parser = argparse.ArgumentParser(description="Run ICA diagnostics on filtered checkpoint.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--subject", required=True, help="sub-001 or 1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = resolve_datasets_arg(args.dataset)[0]
    subject_id = args.subject if args.subject.startswith("sub-") else f"sub-{int(args.subject):03d}"
    subject_num = subject_num_from_id(subject_id)
    participant_id = f"sub-{subject_num:03d}"

    config = load_experiment(args.experiment)
    filtered_path = resolve_checkpoint_path(ds.name, args.experiment, participant_id, "filtered")
    if not filtered_path.exists():
        raise SystemExit(
            f"Filtered checkpoint not found: {filtered_path}\n"
            "Run preprocess_dataset.py first."
        )

    filtered = load_checkpoint(filtered_path, "filtered")
    _, meta = stage_ica(filtered, config)
    print(json.dumps(meta, indent=2, default=str))
