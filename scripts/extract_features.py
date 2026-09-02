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
    list_preprocessed_subjects,
    merge_features_parquet,
    prepare_feature_rows,
    read_json,
    sha256_file,
    write_json,
)
from eeg.paths import (
    feature_partitions_dir,
    features_parquet_path,
    resolve_checkpoint_path,
    subject_log_path,
)
from eeg.runner import run_parallel, summarize_batch, update_experiment_metadata


def _partition_path(dataset_name, experiment, participant_id) -> Path:
    return feature_partitions_dir(dataset_name, experiment) / f"{participant_id}.parquet"


def _subject_done(dataset_name, experiment, participant_id, epochs_path, config_fp) -> bool:
    log_path = subject_log_path(dataset_name, experiment, participant_id, stage="features")
    partition_path = _partition_path(dataset_name, experiment, participant_id)
    if not log_path.exists() or not partition_path.is_file():
        return False
    try:
        log = read_json(log_path)
    except (OSError, ValueError):
        return False
    if log.get("config_fingerprint") != config_fp:
        return False
    if not epochs_path.exists():
        return False
    try:
        epochs_hash = sha256_file(epochs_path)
    except OSError:
        return False
    return (
        log.get("epochs_sha256") == epochs_hash
        and log.get("status") == "ok"
        and log.get("partition") == partition_path.name
    )


def _write_partition(frame: pd.DataFrame, path: Path) -> None:
    """Atomically commit one subject so a Kaggle timeout cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        frame.to_parquet(tmp_path, engine="pyarrow", index=False)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _adopt_legacy_partitions(
    dataset_name: str,
    experiment: str,
    participants: pd.DataFrame,
    config_fp: str,
) -> int:
    """Recover completed partitions left by the pre-checkpoint implementation."""
    participant_ids = set(participants["participant_id"].astype(str))
    adopted = 0
    feature_root = features_parquet_path(dataset_name, experiment).parent
    for legacy_path in sorted(feature_root.glob(".feature-parts-*/*.parquet")):
        participant_id = legacy_path.stem
        destination = _partition_path(dataset_name, experiment, participant_id)
        if participant_id not in participant_ids or destination.exists():
            continue
        epochs_path = resolve_checkpoint_path(
            dataset_name, experiment, participant_id, "epochs"
        )
        if not epochs_path.is_file():
            continue
        try:
            frame = pd.read_parquet(legacy_path)
        except Exception:
            continue
        if (
            frame.empty
            or "participant_id" not in frame.columns
            or set(frame["participant_id"].astype(str)) != {participant_id}
        ):
            continue
        _write_partition(frame, destination)
        write_json(
            subject_log_path(
                dataset_name, experiment, participant_id, stage="features"
            ),
            {
                "participant_id": participant_id,
                "status": "ok",
                "config_fingerprint": config_fp,
                "epochs_sha256": sha256_file(epochs_path),
                "n_epochs": len(frame),
                "runtime_seconds": None,
                "partition": destination.name,
                "recovered_from_legacy_partition": True,
            },
        )
        adopted += 1
    return adopted


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
        _write_partition(feature_rows, partition_path)
        log = {
            "participant_id": participant_id,
            "status": "ok",
            "config_fingerprint": config_fp,
            "epochs_sha256": sha256_file(epochs_path),
            "n_epochs": len(features),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
            "partition": partition_path.name,
        }
        # The partition is committed first. A success log therefore certifies a
        # complete, readable checkpoint and is safe to copy into the next run.
        write_json(
            subject_log_path(
                dataset_name, experiment, participant_id, stage="features"
            ),
            log,
        )
        return {**log, "_partition_path": str(partition_path)}
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
        participants = list_preprocessed_subjects(ds, experiment)
        if subject:
            participants = participants[participants["participant_id"] == subject]

        adopted = _adopt_legacy_partitions(
            ds.name, experiment, participants, config_fp
        )
        if adopted:
            print(
                f"[{ds.name}] recovered {adopted} completed subject partitions "
                "from the timed-out legacy run",
                flush=True,
            )

        output_path = features_parquet_path(ds.name, experiment)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tasks = []
        skipped = []
        for _, row in participants.iterrows():
            participant_id = row["participant_id"]
            epochs_path = resolve_checkpoint_path(
                ds.name, experiment, participant_id, "epochs"
            )
            if not force and _subject_done(
                ds.name, experiment, participant_id, epochs_path, config_fp
            ):
                skipped.append({"participant_id": participant_id, "status": "skipped"})
            else:
                tasks.append(
                    (
                        ds.name,
                        ds.id,
                        experiment,
                        participant_id,
                        row["Group"],
                        config_fp,
                        force,
                        str(_partition_path(ds.name, experiment, participant_id)),
                    )
                )
        if limit is not None:
            tasks = tasks[:limit]

        print(
            f"[{ds.name}] total={len(participants)} already_checkpointed={len(skipped)} "
            f"running_now={len(tasks)}",
            flush=True,
        )
        results = skipped + run_parallel(tasks, _extract_worker, workers=workers)

        # Rebuild the canonical table from every valid durable partition. This
        # makes the aggregate usable after each bounded run, while downstream
        # contracts still reject it until the full cohort is present.
        valid_frames = []
        for _, row in participants.iterrows():
            participant_id = row["participant_id"]
            epochs_path = resolve_checkpoint_path(
                ds.name, experiment, participant_id, "epochs"
            )
            part = _partition_path(ds.name, experiment, participant_id)
            if _subject_done(
                ds.name, experiment, participant_id, epochs_path, config_fp
            ):
                valid_frames.append(pd.read_parquet(part))
        if valid_frames:
            merge_features_parquet(valid_frames, ds.name, experiment)

        batch = summarize_batch(
            [type("R", (), {"status": r.get("status", "ok"), "log": r})() for r in results]
        )
        remaining = len(participants) - len(valid_frames)
        update_experiment_metadata(
            ds.name,
            experiment,
            config,
            "features",
            batch,
            len(participants),
            extra_metadata={
                "checkpointed_subjects": len(valid_frames),
                "remaining_subjects": remaining,
                "batch_limit": limit,
            },
        )
        all_results.extend(results)
        print(
            f"[{ds.name}] completed_now={batch.completed} failed_now={batch.failed} "
            f"checkpointed={len(valid_frames)}/{len(participants)} remaining={remaining}",
            flush=True,
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
