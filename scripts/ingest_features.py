#!/usr/bin/env python3
"""Deprecated: use preprocess_dataset.py + extract_features.py instead."""

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg.config import resolve_dataset
from scripts.extract_features import run_extract
from scripts.preprocess_dataset import run_preprocess

warnings.warn(
    "ingest_features.py is deprecated; use preprocess_dataset.py and extract_features.py",
    DeprecationWarning,
    stacklevel=2,
)


def parse_args():
    parser = argparse.ArgumentParser(description="[DEPRECATED] Ingest raw EEG and extract features.")
    parser.add_argument("--dataset", type=int, action="append", help="Dataset ID (2 or 3).")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--subject", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fast", action="store_true", help="Use experiments/fast.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _dataset_tag(args) -> str:
    if args.all_datasets:
        return "all"
    if args.dataset:
        return str(args.dataset[0]) if len(args.dataset) == 1 else "all"
    return "all"


if __name__ == "__main__":
    args = parse_args()
    experiment = args.experiment or ("fast" if args.fast else "baseline")
    tag = _dataset_tag(args)

    subject = f"sub-{args.subject:03d}" if args.subject else None
    run_preprocess(tag, experiment, workers=args.workers, force=args.force, limit=args.limit, subject=subject)
    if args.all or args.all_datasets or args.subject is None:
        run_extract(tag, experiment, workers=args.workers, force=args.force, limit=args.limit, subject=subject)
    else:
        run_extract(tag, experiment, workers=args.workers, force=args.force, subject=subject)
