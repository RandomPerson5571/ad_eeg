#!/usr/bin/env python3
"""Compare preprocessing QC metrics across experiments."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg.preprocess_report import compare_experiments
from eeg.paths import qc_report_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Compare preprocessing across experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--experiments",
        required=True,
        help="Comma-separated experiment names, e.g. baseline,strict,aggressive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: data/preprocessed/{dataset}/comparison.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    out = args.output
    if out is None:
        out = qc_report_dir(args.dataset, experiments[0]).parent / "experiment_comparison.csv"
    df = compare_experiments(args.dataset, experiments, output_path=out)
    print(df.to_string(index=False))
    print(f"\nWrote: {out}")
