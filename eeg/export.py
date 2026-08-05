"""Export preprocessed epochs to .npy for downstream DL work."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from eeg.io import list_subjects, load_checkpoint
from eeg.paths import checkpoint_path, epochs_npy_dir


def _normalize_epochs(data: np.ndarray, method: str | None) -> np.ndarray:
    if method is None:
        return data.astype(np.float32)
    if method == "zscore":
        mean = data.mean(axis=(0, 2), keepdims=True)
        std = data.std(axis=(0, 2), keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return ((data - mean) / std).astype(np.float32)
    raise ValueError(f"Unknown normalization: {method}")


def export_epochs_npy(
    dataset: str,
    experiment: str,
    participant_id: str,
    normalize: str | None = None,
) -> Path:
    """Export one subject's epochs to (n_epochs, n_channels, n_samples) float32 .npy."""
    epochs_path = checkpoint_path(dataset, experiment, participant_id, "epochs")
    if not epochs_path.exists():
        raise FileNotFoundError(f"Missing epochs checkpoint: {epochs_path}")

    epochs = load_checkpoint(epochs_path, "epochs")
    data = epochs.get_data()
    data = _normalize_epochs(data, normalize)

    out_dir = epochs_npy_dir(dataset, experiment)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{participant_id}.npy"
    np.save(out_path, data)
    return out_path


def export_all_epochs_npy(
    dataset: str,
    experiment: str,
    subjects: Iterable[str] | None = None,
    normalize: str | None = None,
) -> list[Path]:
    """Batch-export epoch arrays for all (or selected) subjects."""
    from eeg.config import resolve_dataset

    spec = resolve_dataset(dataset)[0]
    participants = list_subjects(spec)
    if subjects is not None:
        subject_set = set(subjects)
        participants = participants[participants["participant_id"].isin(subject_set)]

    paths: list[Path] = []
    for _, row in participants.iterrows():
        pid = row["participant_id"]
        try:
            paths.append(export_epochs_npy(dataset, experiment, pid, normalize=normalize))
        except FileNotFoundError:
            continue
    return paths
