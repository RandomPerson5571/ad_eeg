#!/usr/bin/env python3
"""Extract features from preprocessed epoch checkpoints."""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg.cli import add_common_args, parse_subject_arg, resolve_datasets_arg
from eeg.config import config_fingerprint, load_experiment
from eeg.features import extract_from_epochs
from eeg.io import (
    load_checkpoint,
    list_subjects,
    merge_features_parquet,
    prepare_feature_rows,
    read_json,
    sha256_file,
    write_json,
)
from eeg.paths import features_parquet_path, resolve_checkpoint_path, subject_log_path
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


def _extract_worker(
    dataset_name,
    dataset_id,
    experiment,
    participant_id,
    label,
    config_fp,
    force,
    partition_path,
):
    epochs_path = resolve_checkpoint_path(
        dataset_name,
        experiment,
        participant_id,
        "epochs",
    )

    if not force and _subject_done(
        dataset_name,
        experiment,
        participant_id,
        epochs_path,
        config_fp,
    ):
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
        feature_rows = prepare_feature_rows(
            features,
            participant_id,
            dataset_name,
            label,
            dataset_id,
        )
        partition_path = Path(partition_path)
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        feature_rows.to_parquet(partition_path, engine="pyarrow", index=False)
        log = {
            "participant_id": participant_id,
            "status": "ok",
            "config_fingerprint": config_fp,
            "epochs_sha256": sha256_file(epochs_path),
            "n_epochs": len(features),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
            "_partition_path": str(partition_path),
        }
        return log
    except Exception as exc:
        log = {
            "participant_id": participant_id,
            "status": "error",
            "error": str(exc),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
        }
        write_json(
            subject_log_path(
                dataset_name,
                experiment,
                participant_id,
                stage="features",
            ),
            log,
        )
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

        output_path = features_parquet_path(ds.name, experiment)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".feature-parts-",
            dir=output_path.parent,
        ) as partition_dir:
            tasks = []
            for _, row in participants.iterrows():
                participant_id = row["participant_id"]
                tasks.append(
                    (
                        ds.name,
                        ds.id,
                        experiment,
                        participant_id,
                        row["Group"],
                        config_fp,
                        force,
                        str(Path(partition_dir) / f"{participant_id}.parquet"),
                    )
                )
            if limit:
                tasks = tasks[:limit]

            results = run_parallel(tasks, _extract_worker, workers=workers)
            completed = [result for result in results if result.get("status") == "ok"]
            if completed:
                frames = [
                    pd.read_parquet(result["_partition_path"])
                    for result in completed
                ]
                merge_features_parquet(frames, ds.name, experiment)
                for result in completed:
                    result.pop("_partition_path", None)
                    write_json(
                        subject_log_path(
                            ds.name,
                            experiment,
                            result["participant_id"],
                            stage="features",
                        ),
                        result,
                    )
        batch = summarize_batch(
            [type("R", (), {"status": r.get("status", "ok"), "log": r})() for r in results]
        )
        update_experiment_metadata(
            ds.name,
            experiment,
            config,
            "features",
            batch,
            len(tasks),
        )
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
