#!/usr/bin/env python3
"""Inspect preprocessing for a single subject (QC metrics + plots)."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

from eeg.cli import resolve_datasets_arg, subject_num_from_id
from eeg.qc import preprocessing_metrics
from eeg.visualization import plot_preprocessing_panels


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect preprocessing for one subject.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--subject", required=True, help="sub-001 or 1")
    parser.add_argument("--output", type=Path, help="Output directory for plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = resolve_datasets_arg(args.dataset)[0]
    subject_id = args.subject if args.subject.startswith("sub-") else f"sub-{int(args.subject):03d}"
    subject_num = subject_num_from_id(subject_id)

    metrics = preprocessing_metrics(ds, subject_num, args.experiment)
    out = plot_preprocessing_panels(ds, subject_num, args.experiment, args.output)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved QC report: {out}")
