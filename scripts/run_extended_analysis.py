#!/usr/bin/env python3
"""Run the complete leakage-safe follow-up analysis suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg.config import load_experiment
from eeg.repro import init_repro, snapshot_environment
from eeg.training.extended_analysis import run_extended_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run threshold/calibration, LR regularization, feature ablations, "
            "feature selection, and a subject-label permutation test."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument(
        "--permutations",
        type=int,
        default=None,
        help="Number of label permutations (default: configs/training.yaml or 1000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_experiment(args.experiment)
    config["seed"] = config.get("training", {}).get("random_state", 42)
    init_repro(config["seed"])
    result = run_extended_analysis(
        dataset=args.dataset,
        experiment=args.experiment,
        config=config,
        cv_folds=args.cv_folds,
        n_permutations=args.permutations,
    )
    print(f"Wrote {result['analysis_summary']}")
    print(f"Wrote {result['permutation_scores']}")
    print(f"Wrote {result['report']}")
    print(snapshot_environment())
