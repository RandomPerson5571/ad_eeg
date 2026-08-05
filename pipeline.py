#!/usr/bin/env python3
"""Orchestrate pipeline stages: preprocess → features → train."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_features import run_extract
from scripts.preprocess_dataset import run_preprocess
from eeg.training import train_models
from eeg.cli import resolve_datasets_arg
from eeg.config import load_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run EEG pipeline stages.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument(
        "--stages",
        default="preprocess,features,train",
        help="Comma-separated: preprocess,features,train",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="xgboost,mlp")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stages = {s.strip() for s in args.stages.split(",")}
    config = load_experiment(args.experiment)
    models = [m.strip() for m in args.model.split(",") if m.strip()]

    if "preprocess" in stages:
        run_preprocess(
            args.dataset,
            args.experiment,
            workers=args.workers,
            force=args.force,
            limit=args.limit,
        )

    if "features" in stages:
        run_extract(
            args.dataset,
            args.experiment,
            workers=args.workers,
            force=args.force,
            limit=args.limit,
        )

    if "train" in stages:
        for ds in resolve_datasets_arg(args.dataset):
            train_models(ds.name, args.experiment, ds.id, models, config)
