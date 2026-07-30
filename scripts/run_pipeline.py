#!/usr/bin/env python3
"""End-to-end pipeline: ingest features, train models, save metrics."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS, RESULTS_DIR  # noqa: E402
from scripts.ingest_features import run_ingest  # noqa: E402


def save_preprocessing_config():
    from config import (
        EPOCH_LENGTH,
        EPOCH_OVERLAP,
        FEATURE_COLUMNS,
        RANDOM_STATE,
        SAMPLING_RATE,
        TEST_SIZE,
    )

    snapshot = {
        "SAMPLING_RATE": SAMPLING_RATE,
        "EPOCH_LENGTH": EPOCH_LENGTH,
        "EPOCH_OVERLAP": EPOCH_OVERLAP,
        "FEATURE_COLUMNS": FEATURE_COLUMNS,
        "TEST_SIZE": TEST_SIZE,
        "RANDOM_STATE": RANDOM_STATE,
        "preprocessing": {
            "bandpass_hz": [0.5, 40],
            "autoreject": True,
            "input_source": "raw",
        },
    }
    path = Path(RESULTS_DIR) / "preprocessing_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Preprocessing config saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full EEG analysis pipeline.")
    parser.add_argument("--ingest", action="store_true", help="Run feature ingestion.")
    parser.add_argument("--all-datasets", action="store_true", help="Ingest all configured datasets.")
    parser.add_argument("--limit", type=int, help="Limit subjects per dataset during ingest.")
    parser.add_argument("--train", type=str, help="Comma-separated models: xgboost,mlp")
    parser.add_argument("--subject-split", action="store_true", help="Use subject-level splits (default in train_utils).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.ingest and not args.train:
        print("Nothing to do. Pass --ingest and/or --train xgboost,mlp")
        sys.exit(1)

    if args.ingest:
        datasets = DATASETS if args.all_datasets else DATASETS
        run_ingest(datasets=datasets, subject_nums=None, limit=args.limit)
        save_preprocessing_config()

    if args.train:
        models = [m.strip().lower() for m in args.train.split(",")]
        all_metrics = {}

        if "xgboost" in models:
            from classifier_models.train_XGBoost import train_xgboost

            _, metrics = train_xgboost()
            all_metrics["xgboost"] = metrics

        if "mlp" in models:
            from classifier_models.train_mlp import train_mlp

            _, _, metrics = train_mlp()
            all_metrics["mlp"] = metrics

        out = Path(RESULTS_DIR) / "metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Combined metrics saved to {out}")
