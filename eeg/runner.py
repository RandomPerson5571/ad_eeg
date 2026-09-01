"""Parallel batch runner with auto-resume."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from eeg.config import experiment_metadata, load_experiment
from eeg.io import write_json
from eeg.paths import experiment_metadata_path


@dataclass
class BatchResult:
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    entries: list[dict[str, Any]] | None = None

    def __post_init__(self):
        if self.entries is None:
            self.entries = []


def run_parallel(
    tasks: list[tuple],
    worker_fn: Callable,
    workers: int = 1,
) -> list[Any]:
    if workers <= 1 or len(tasks) <= 1:
        return [worker_fn(*task) for task in tasks]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_fn, *task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def update_experiment_metadata(
    dataset_name: str,
    experiment_name: str,
    config: dict,
    stage: str,
    batch: BatchResult,
    n_subjects: int,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    path = experiment_metadata_path(dataset_name, experiment_name, stage)
    meta = experiment_metadata(
        dataset_name,
        experiment_name,
        config,
        n_subjects=n_subjects,
        n_completed=batch.completed,
        n_skipped=batch.skipped,
        n_failed=batch.failed,
        **(extra_metadata or {}),
    )
    write_json(path, meta)
    return path


def summarize_batch(entries: list) -> BatchResult:
    batch = BatchResult()
    for entry in entries:
        status = getattr(entry, "status", None) or entry.get("status", "ok")
        if status == "skipped":
            batch.skipped += 1
        elif status == "error":
            batch.failed += 1
        else:
            batch.completed += 1
        log = getattr(entry, "log", None) or entry
        batch.entries.append(log if isinstance(log, dict) else {"status": status})
    return batch
