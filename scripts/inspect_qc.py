#!/usr/bin/env python3
"""Generate QC reports for manual inspection of preprocessing and features."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS, PARQUET_COMBINED_FILE  # noqa: E402
from util.qc import (  # noqa: E402
    feature_summary,
    plot_feature_report,
    plot_preprocessing_report,
    spot_check_features,
)


def cmd_preprocess(args):
    out = plot_preprocessing_report(
        dataset_id=args.dataset,
        subject_num=args.subject,
        output_dir=Path(args.output) if args.output else None,
        show=args.show,
    )
    summary_path = out.with_suffix(".json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved preprocessing QC report: {out}")


def cmd_features(args):
    out = plot_feature_report(
        output_dir=Path(args.output) if args.output else None,
        show=args.show,
    )
    summary = feature_summary()
    print(json.dumps({k: v for k, v in summary.items() if k != "features_by_label"}, indent=2))
    print(f"\nSaved feature QC report: {out}")


def cmd_spot_check(args):
    comparison = spot_check_features(args.dataset, args.subject)
    out_path = Path(args.output) if args.output else PROJECT_ROOT / "results" / "qc" / "features" / f"spot_check_dataset{args.dataset}_sub{args.subject:03d}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_path, index=False)
    print(f"Spot-check saved to {out_path}")
    print(comparison.groupby("source")[["epoch_id"]].count())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual QC for preprocessing and feature extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inspect_qc.py preprocess --dataset 2 --subject 1
  python scripts/inspect_qc.py features
  python scripts/inspect_qc.py spot-check --dataset 2 --subject 1
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="QC plots for one subject's preprocessing")
    p_pre.add_argument("--dataset", type=int, default=2, choices=DATASETS)
    p_pre.add_argument("--subject", type=int, default=1)
    p_pre.add_argument("--output", type=str, help="Output directory (default: results/qc/preprocessing/)")
    p_pre.add_argument("--show", action="store_true", help="Display plots interactively")
    p_pre.set_defaults(func=cmd_preprocess)

    p_feat = sub.add_parser("features", help="QC plots for the feature parquet store")
    p_feat.add_argument("--output", type=str, help="Output directory (default: results/qc/features/)")
    p_feat.add_argument("--show", action="store_true")
    p_feat.set_defaults(func=cmd_features)

    p_spot = sub.add_parser("spot-check", help="Re-extract one subject and compare to parquet")
    p_spot.add_argument("--dataset", type=int, default=2, choices=DATASETS)
    p_spot.add_argument("--subject", type=int, default=1)
    p_spot.add_argument("--output", type=str)
    p_spot.set_defaults(func=cmd_spot_check)

    return parser.parse_args()


if __name__ == "__main__":
    if not Path(PARQUET_COMBINED_FILE).parent.exists():
        Path(PARQUET_COMBINED_FILE).parent.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    args.func(args)
