#!/usr/bin/env python3
"""CLI wrapper around eeg.training.benchmark.run_benchmark."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg.config import load_experiment
from eeg.repro import init_repro, snapshot_environment
from eeg.training.benchmark import run_benchmark


def parse_args():
    p = argparse.ArgumentParser(description="Run classical ML benchmark.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--experiment", default="baseline")
    p.add_argument(
        "--models",
        default="logistic_regression,random_forest,xgboost,mlp",
        help="Comma-separated model keys",
    )
    p.add_argument("--cv-folds", type=int, default=None)
    return p.parse_args()


def build_config(experiment: str) -> dict:
    cfg = load_experiment(experiment)
    cfg["seed"] = cfg.get("training", {}).get("random_state", 42)
    cfg["dataset_name"] = None
    cfg["experiment_name"] = experiment
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args.experiment)
    config["dataset_name"] = args.dataset
    init_repro(config["seed"])
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    result = run_benchmark(
        dataset=args.dataset,
        experiment=args.experiment,
        models=models,
        config=config,
        cv_folds=args.cv_folds,
    )
    print(f"Wrote {result['benchmark_csv']}")
    print(snapshot_environment())
