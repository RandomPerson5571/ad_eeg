#!/usr/bin/env python3
"""Extract features from preprocessed epoch checkpoints."""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg.cli import add_common_args, parse_subject_arg, resolve_datasets_arg
from eeg.config import config_fingerprint, load_experiment
from eeg.features import extract_from_epochs
from eeg.io import (
    load_checkpoint,
    list_subjects,
    read_json,
    save_features_parquet,
    sha256_file,
    write_json,
)
from eeg.paths import checkpoint_path, features_parquet_path, subject_log_path
from eeg.runner import run_parallel, summarize_batch, update_experiment_metadata


def _subject_done(dataset_name, experiment, participant_id, epochs_path, config_fp) -> bool:
    log_path = subject_log_path(dataset_name, experiment, participant_id, stage="features")
    if not log_path.exists():
        return False
    log = read_json(log_path)
    if log.get("config_fingerprint") != config_fp:
        return False
    if not epochs_path.exists():
        return False
    try:
        epochs_hash = sha256_file(epochs_path)
    except OSError:
        return False
    return log.get("epochs_sha256") == epochs_hash and log.get("status") == "ok"


def _extract_worker(args_tuple):
    dataset_name, dataset_id, experiment, participant_id, label, config, config_fp, force = args_tuple
    epochs_path = checkpoint_path(dataset_name, experiment, participant_id, "epochs")

    if not force and _subject_done(dataset_name, experiment, participant_id, epochs_path, config_fp):
        return {"participant_id": participant_id, "status": "skipped"}

    if not epochs_path.exists():
        return {
            "participant_id": participant_id,
            "status": "error",
            "error": f"Missing epochs checkpoint: {epochs_path}",
        }

    t0 = time.perf_counter()
    try:
        epochs = load_checkpoint(epochs_path, "epochs")
        features = extract_from_epochs(epochs)
        save_features_parquet(features, participant_id, dataset_name, experiment, label, dataset_id)
        log = {
            "participant_id": participant_id,
            "status": "ok",
            "config_fingerprint": config_fp,
            "epochs_sha256": sha256_file(epochs_path),
            "n_epochs": len(features),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
        }
        write_json(subject_log_path(dataset_name, experiment, participant_id, stage="features"), log)
        return log
    except Exception as exc:
        log = {
            "participant_id": participant_id,
            "status": "error",
            "error": str(exc),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
        }
        write_json(subject_log_path(dataset_name, experiment, participant_id, stage="features"), log)
        return log


def run_extract(
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
        participants = list_subjects(ds)
        if subject:
            participants = participants[participants["participant_id"] == subject]

        tasks = []
        for _, row in participants.iterrows():
            tasks.append(
                (ds.name, ds.id, experiment, row["participant_id"], row["Group"], config, config_fp, force)
            )
        if limit:
            tasks = tasks[:limit]

        features_parquet_path(ds.name, experiment).parent.mkdir(parents=True, exist_ok=True)
        results = run_parallel(tasks, _extract_worker, workers=workers)
        batch = summarize_batch(
            [type("R", (), {"status": r.get("status", "ok"), "log": r})() for r in results]
        )
        update_experiment_metadata(ds.name, experiment, config, "features", batch, len(tasks))
        all_results.extend(results)
        print(
            f"[{ds.name}] completed={batch.completed} skipped={batch.skipped} failed={batch.failed}"
        )

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from preprocessed epochs.")
    add_common_args(parser)
    parser.add_argument("--subject", help="Single subject: sub-001 or 1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    subject = parse_subject_arg(args.subject) if args.subject else None
    run_extract(
        dataset_tag=args.dataset,
        experiment=args.experiment,
        workers=args.workers,
        force=args.force,
        limit=args.limit,
        subject=subject,
    )
