"""Artifact contracts shared by the staged Kaggle pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from eeg.config import resolve_dataset
from eeg.io import configured_participant_index
from eeg.paths import (
    feature_importance_path,
    features_parquet_path,
    preprocessed_dir,
    results_dir,
    selected_features_path,
)


class ArtifactContractError(ValueError):
    """A pipeline artifact exists but cannot be consumed by the next stage."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {produced_by} artifact is missing: {path}")
    return path


def validate_preprocessed_artifacts(
    dataset: str,
    experiment: str,
    *,
    require_participants: bool = True,
) -> dict:
    """Validate notebook 01 output consumed by notebooks 02 and 03."""
    root = preprocessed_dir(dataset, experiment)
    epoch_files = sorted(root.glob("sub-*_epo.fif"))
    if not epoch_files:
        raise FileNotFoundError(
            f"No epoch checkpoints found in {root}. Attach the full-mode output "
            "from notebook 01 and set PIPELINE_INPUT to that dataset."
        )

    metadata_path = root / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    participant_index = metadata.get("participant_index", [])
    participants = pd.DataFrame(participant_index)
    participant_source = "metadata.json"
    if participants.empty:
        dataset_spec = resolve_dataset(dataset)[0]
        participants = configured_participant_index(dataset_spec)
        participant_source = "configs/dataset.yaml"
    if participants.empty and not require_participants:
        return {
            "root": root,
            "participants": None,
            "epoch_checkpoints": len(epoch_files),
        }
    required = {"participant_id", "Group"}
    missing = sorted(required - set(participants.columns))
    if missing:
        raise ArtifactContractError(
            f"{metadata_path} participant_index is missing columns required for "
            f"feature extraction: {missing}"
        )
    if participants.empty or participants["participant_id"].duplicated().any():
        raise ArtifactContractError(
            f"{metadata_path} participant_index must contain unique participant_id rows."
        )

    checkpoint_ids = {p.name[: -len("_epo.fif")] for p in epoch_files}
    indexed_ids = set(participants["participant_id"].astype(str))
    unknown = sorted(checkpoint_ids - indexed_ids)
    if unknown:
        raise ArtifactContractError(
            f"Epoch checkpoints are absent from {metadata_path} participant_index: "
            f"{unknown[:5]}"
        )
    if participant_source == "configs/dataset.yaml":
        participants = participants[
            participants["participant_id"].astype(str).isin(checkpoint_ids)
        ]
    return {
        "root": root,
        "participants": len(participants),
        "participant_source": participant_source,
        "epoch_checkpoints": len(epoch_files),
    }


def validate_epoch_array(
    data: np.ndarray, source: str | Path = "epoch array"
) -> tuple[int, int, int]:
    """Require the DL interchange shape: epochs x channels x samples, float32."""
    source = str(source)
    if data.ndim != 3:
        raise ArtifactContractError(
            f"{source} must have shape (n_epochs, n_channels, n_samples); got {data.shape}."
        )
    if any(size <= 0 for size in data.shape):
        raise ArtifactContractError(
            f"{source} contains an empty dimension: {data.shape}."
        )
    if data.dtype != np.float32:
        raise ArtifactContractError(f"{source} must be float32; got {data.dtype}.")
    return tuple(int(size) for size in data.shape)


def validate_epoch_exports(paths: Iterable[str | Path]) -> dict:
    """Validate notebook 02 outputs and cross-subject channel/sample compatibility."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise FileNotFoundError(
            "No exported epoch .npy files were produced. The notebook 01 artifact "
            "must contain data/preprocessed/<dataset>/<experiment>/*_epo.fif."
        )

    shapes: dict[str, tuple[int, int, int]] = {}
    expected_tail: tuple[int, int] | None = None
    for path in paths:
        _require_file(path, "notebook 02 epoch-export")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = validate_epoch_array(array, path)
        tail = shape[1:]
        if expected_tail is None:
            expected_tail = tail
        elif tail != expected_tail:
            raise ArtifactContractError(
                f"Incompatible epoch shape in {path}: channels/samples {tail}, "
                f"expected {expected_tail}."
            )
        shapes[path.name] = shape

    return {
        "subjects": len(paths),
        "n_channels": expected_tail[0],
        "n_samples": expected_tail[1],
        "dtype": "float32",
        "shapes": shapes,
    }


def validate_feature_artifact(
    dataset: str,
    experiment: str,
    required_features: Iterable[str],
) -> dict:
    """Validate notebook 03 output consumed by selection and training."""
    required_features = list(required_features)
    path = _require_file(
        features_parquet_path(dataset, experiment), "notebook 03 feature"
    )
    frame = pd.read_parquet(path)
    required = {"participant_id", "label", *required_features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ArtifactContractError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ArtifactContractError(f"{path} contains no feature rows.")
    if frame["participant_id"].isna().any() or frame["label"].isna().any():
        raise ArtifactContractError(
            f"{path} contains missing participant IDs or labels."
        )
    return {
        "path": path,
        "rows": len(frame),
        "subjects": frame["participant_id"].nunique(),
        "features": len(required_features),
    }


def validate_selection_artifacts(dataset: str, experiment: str) -> dict:
    """Validate notebook 04 output consumed by model training."""
    selected_path = _require_file(
        selected_features_path(dataset, experiment), "notebook 04 selected-feature"
    )
    importance_path = _require_file(
        feature_importance_path(dataset, experiment), "notebook 04 importance"
    )
    selected = pd.read_parquet(selected_path)
    importance = pd.read_csv(importance_path)
    identity = {"participant_id", "label"}
    missing = sorted(identity - set(selected.columns))
    identity_columns = {
        "participant_id",
        "label",
        "dataset_id",
        "dataset_name",
        "epoch_id",
    }
    feature_cols = [
        column for column in selected.columns if column not in identity_columns
    ]
    if missing or not feature_cols:
        raise ArtifactContractError(
            f"{selected_path} needs participant_id, label, and at least one selected "
            f"feature; missing={missing}."
        )
    if not {"feature", "mi_score"}.issubset(importance.columns):
        raise ArtifactContractError(
            f"{importance_path} must contain feature and mi_score columns."
        )
    return {"rows": len(selected), "selected_features": len(feature_cols)}


def validate_benchmark_artifacts(dataset: str, experiment: str) -> dict:
    """Validate model benchmark output consumed by final aggregation."""
    root = results_dir(dataset, experiment)
    benchmark_path = _require_file(root / "benchmark.csv", "benchmark")
    predictions_path = _require_file(root / "predictions.csv", "benchmark prediction")
    benchmark = pd.read_csv(benchmark_path)
    predictions = pd.read_csv(predictions_path)
    benchmark_required = {"model", "dataset", "experiment", "balanced_accuracy"}
    prediction_required = {"model", "participant_id", "y_true", "y_pred"}
    missing_benchmark = sorted(benchmark_required - set(benchmark.columns))
    missing_predictions = sorted(prediction_required - set(predictions.columns))
    if benchmark.empty or missing_benchmark:
        raise ArtifactContractError(
            f"{benchmark_path} is empty or missing columns: {missing_benchmark}"
        )
    if predictions.empty or missing_predictions:
        raise ArtifactContractError(
            f"{predictions_path} is empty or missing columns: {missing_predictions}"
        )
    return {"models": len(benchmark), "subject_predictions": len(predictions)}
