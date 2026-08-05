#!/usr/bin/env python3
"""Deprecated: use pipeline.py instead."""

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.warn(
    "run_pipeline.py is deprecated; use pipeline.py",
    DeprecationWarning,
    stacklevel=2,
)

from eeg.config import resolve_dataset, load_experiment
from eeg.training import train_models
from scripts.extract_features import run_extract
from scripts.preprocess_dataset import run_preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="[DEPRECATED] Run the full EEG analysis pipeline.")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train", type=str)
    parser.add_argument("--dataset", type=int, help="Dataset ID (2 or 3)")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tag = "all" if args.all_datasets else str(args.dataset or 2)
    config = load_experiment(args.experiment)

    if args.ingest:
        run_preprocess(tag, args.experiment, workers=args.workers, limit=args.limit)
        run_extract(tag, args.experiment, workers=args.workers, limit=args.limit)

    if args.train:
        if args.dataset is None and not args.all_datasets:
            print("Error: --dataset is required when using --train")
            sys.exit(1)
        datasets = resolve_dataset("all" if args.all_datasets else str(args.dataset))
        models = [m.strip() for m in args.train.split(",")]
        for ds in datasets:
            train_models(ds.name, args.experiment, ds.id, models, config)
