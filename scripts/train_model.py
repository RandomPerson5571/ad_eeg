#!/usr/bin/env python3
"""Train classifiers on extracted features (includes ROC + confusion matrix)."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg.cli import resolve_datasets_arg
from eeg.config import load_experiment
from eeg.training import train_models


def parse_args():
    parser = argparse.ArgumentParser(description="Train models on extracted features.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--model", default="xgboost,mlp", help="Comma-separated: xgboost,mlp")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_experiment(args.experiment)
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    datasets = resolve_datasets_arg(args.dataset)

    for ds in datasets:
        print(f"Training on {ds.name} (experiment={args.experiment})...")
        results = train_models(ds.name, args.experiment, ds.id, models, config)
        print(json.dumps(results, indent=2))
