"""Backward-compatible entry point. Prefer: python scripts/ingest_features.py"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS
from scripts.ingest_features import parse_args, run_ingest

if __name__ == "__main__":
    args = parse_args()
    if args.all_datasets:
        datasets = DATASETS
    elif args.dataset:
        datasets = args.dataset
    else:
        datasets = DATASETS

    if args.subject:
        subject_nums = [args.subject]
    elif args.all:
        subject_nums = None
    else:
        subject_nums = [1]

    run_ingest(
        datasets=datasets,
        subject_nums=subject_nums,
        limit=args.limit,
        save_clean=args.save_clean,
        qc=args.qc,
    )
