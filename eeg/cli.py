"""Shared CLI argument helpers."""

from __future__ import annotations

import argparse
import re

from eeg.config import resolve_dataset


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset tag: eyesclosed, photomark, 2, 3, or all",
    )
    parser.add_argument(
        "--experiment",
        default="baseline",
        help="Experiment config name (loads experiments/{name}.yaml)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoints, full rerun")
    parser.add_argument("--limit", type=int, help="Max subjects per dataset")


def parse_subject_arg(subject: str | None) -> str | None:
    if subject is None:
        return None
    if subject.startswith("sub-"):
        return subject
    if subject.isdigit():
        return f"sub-{int(subject):03d}"
    return subject


def subject_num_from_id(participant_id: str) -> int:
    m = re.search(r"(\d+)$", participant_id)
    if not m:
        raise ValueError(f"Cannot parse subject number from {participant_id}")
    return int(m.group(1))


def resolve_datasets_arg(tag: str):
    return resolve_dataset(tag)
