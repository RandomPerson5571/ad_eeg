#!/usr/bin/env python3
"""Regenerate spectral QC metrics from existing checkpoints (no preprocessing rerun)."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg.cli import add_common_args, resolve_datasets_arg
from eeg.config import load_experiment
from eeg.preprocess_report import write_preprocess_report
from eeg.qc import backfill_spectral_qc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill spectral QC in subject logs from raw+clean checkpoints."
    )
    add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_experiment(args.experiment)

    for ds in resolve_datasets_arg(args.dataset):
        patched = backfill_spectral_qc(ds.name, args.experiment, limit=args.limit)
        print(f"[{ds.name}] patched {len(patched)} subject log(s)")
        if patched:
            print(f"  {', '.join(patched)}")

        report_paths = write_preprocess_report(
            ds.name, args.experiment, config=config, dataset_spec=ds
        )
        print(f"[{ds.name}] QC report: {report_paths['summary_csv']}")
