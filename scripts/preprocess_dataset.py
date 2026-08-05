#!/usr/bin/env python3
"""Preprocess raw EEG into staged checkpoints (raw → filtered → ica → clean → epochs)."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

from eeg.cli import add_common_args, resolve_datasets_arg, subject_num_from_id
from eeg.config import config_fingerprint, load_experiment
from eeg.io import list_subjects
from eeg.paths import preprocessed_dir, raw_eeg_path
from eeg.preprocessing import preprocess_subject
from eeg.runner import run_parallel, summarize_batch, update_experiment_metadata


def _worker(args_tuple):
    raw_path, dataset_name, experiment, participant_id, config, config_fp, force = args_tuple
    return preprocess_subject(
        raw_path=Path(raw_path),
        participant_id=participant_id,
        dataset_name=dataset_name,
        experiment=experiment,
        config=config,
        config_fp=config_fp,
        force=force,
    )


def run_preprocess(
    dataset_tag: str,
    experiment: str,
    workers: int = 1,
    force: bool = False,
    limit: int | None = None,
    subject: str | None = None,
):
    config = load_experiment(experiment)
    config_fp = config_fingerprint(config)
    datasets = resolve_datasets_arg(dataset_tag)
    all_results = []

    for ds in datasets:
        config = load_experiment(experiment)
        config_fp = config_fingerprint(config)

        preprocessed_dir(ds.name, experiment).mkdir(parents=True, exist_ok=True)
        participants = list_subjects(ds)

        if subject:
            participants = participants[participants["participant_id"] == subject]

        tasks = []
        for idx, row in participants.iterrows():
            subject_num = idx + 1
            participant_id = row["participant_id"]
            raw_path = str(raw_eeg_path(ds, subject_num))
            tasks.append(
                (raw_path, ds.name, experiment, participant_id, config, config_fp, force)
            )

        if limit:
            tasks = tasks[:limit]

        results = run_parallel(tasks, _worker, workers=workers)
        batch = summarize_batch(results)
        update_experiment_metadata(ds.name, experiment, config, "preprocessed", batch, len(tasks))
        all_results.extend(results)
        print(
            f"[{ds.name}] completed={batch.completed} skipped={batch.skipped} failed={batch.failed}"
        )

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw EEG with staged checkpoints.")
    add_common_args(parser)
    parser.add_argument("--subject", help="Single subject: sub-001 or 1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    subject = None
    if args.subject:
        from eeg.cli import parse_subject_arg

        subject = parse_subject_arg(args.subject)
    run_preprocess(
        dataset_tag=args.dataset,
        experiment=args.experiment,
        workers=args.workers,
        force=args.force,
        limit=args.limit,
        subject=subject,
    )
