"""Central path resolution for pipeline stages and compat aliases."""

from __future__ import annotations

from pathlib import Path

from eeg.config import PROJECT_ROOT, load_base_configs

STAGES = ("raw", "filtered", "ica", "clean", "epochs")
STAGE_SUFFIX = {
    "raw": "_raw.fif",
    "filtered": "_filtered_raw.fif",
    "ica": "_ica_raw.fif",
    "clean": "_clean_raw.fif",
    "epochs": "_epo.fif",
}
# Pre-MNE-naming checkpoints (resume reads these if new names are absent)
LEGACY_STAGE_SUFFIX = {
    "filtered": "_filtered.fif",
    "ica": "_ica.fif",
    "clean": "_clean.fif",
}


def raw_data_root() -> Path:
    cfg = load_base_configs()
    return PROJECT_ROOT / cfg["dataset"]["raw_data_dir"]


def data_root() -> Path:
    return PROJECT_ROOT / "data"


def experiment_dir(dataset: str, experiment: str, stage: str) -> Path:
    """data/{stage}/{dataset}/{experiment}/"""
    return data_root() / stage / dataset / experiment


def preprocessed_dir(dataset: str, experiment: str) -> Path:
    return experiment_dir(dataset, experiment, "preprocessed")


def qc_report_dir(dataset: str, experiment: str) -> Path:
    return preprocessed_dir(dataset, experiment) / "qc"


def features_dir(dataset: str, experiment: str) -> Path:
    return experiment_dir(dataset, experiment, "features")


def models_dir(dataset: str, experiment: str) -> Path:
    return experiment_dir(dataset, experiment, "models")


def results_dir(dataset: str, experiment: str) -> Path:
    return experiment_dir(dataset, experiment, "results")


def checkpoint_path(
    dataset: str, experiment: str, participant_id: str, stage: str
) -> Path:
    suffix = STAGE_SUFFIX[stage]
    return preprocessed_dir(dataset, experiment) / f"{participant_id}{suffix}"


def resolve_checkpoint_path(
    dataset: str, experiment: str, participant_id: str, stage: str
) -> Path:
    """Return checkpoint path for reading (new MNE naming, else legacy)."""
    primary = checkpoint_path(dataset, experiment, participant_id, stage)
    if primary.exists():
        return primary
    legacy_suffix = LEGACY_STAGE_SUFFIX.get(stage)
    if legacy_suffix:
        legacy = preprocessed_dir(dataset, experiment) / f"{participant_id}{legacy_suffix}"
        if legacy.exists():
            return legacy
    return primary


def subject_log_path(dataset: str, experiment: str, participant_id: str, stage: str = "preprocessed") -> Path:
    return experiment_dir(dataset, experiment, stage) / "logs" / f"{participant_id}.json"


def experiment_metadata_path(dataset: str, experiment: str, stage: str = "preprocessed") -> Path:
    return experiment_dir(dataset, experiment, stage) / "metadata.json"


def features_parquet_path(dataset: str, experiment: str) -> Path:
    return features_dir(dataset, experiment) / "subject_features.parquet"


def legacy_parquet_path(dataset_id: int) -> Path:
    return PROJECT_ROOT / "parquet_files" / f"features_dataset{dataset_id}.parquet"


def legacy_models_dir() -> Path:
    return PROJECT_ROOT / "classifier_models" / "saved_models"


def legacy_results_dir() -> Path:
    return PROJECT_ROOT / "results"


def resolve_features_path(dataset: str, experiment: str, dataset_id: int | None = None) -> Path:
    """Prefer new layout; fall back to legacy parquet if present."""
    new_path = features_parquet_path(dataset, experiment)
    if new_path.exists():
        return new_path
    if dataset_id is not None:
        legacy = legacy_parquet_path(dataset_id)
        if legacy.exists():
            return legacy
    return new_path


def resolve_model_dir(dataset: str, experiment: str) -> Path:
    new_dir = models_dir(dataset, experiment)
    if new_dir.exists() and any(new_dir.iterdir()):
        return new_dir
    legacy = legacy_models_dir()
    if legacy.exists():
        return legacy
    return new_dir


def raw_eeg_path(dataset_spec, subject_num: int) -> Path:
    return (
        dataset_spec.raw_dir
        / f"sub-{subject_num:03d}"
        / "eeg"
        / f"sub-{subject_num:03d}_task-{dataset_spec.task}_eeg.set"
    )


def participants_tsv(dataset_spec) -> Path:
    return dataset_spec.raw_dir / "participants.tsv"


def audit_dir(dataset: str) -> Path:
    return data_root() / "audit" / dataset


def epochs_npy_dir(dataset: str, experiment: str) -> Path:
    return preprocessed_dir(dataset, experiment) / "epochs"


def selected_features_path(dataset: str, experiment: str) -> Path:
    return features_dir(dataset, experiment) / "selected_features.parquet"


def feature_importance_path(dataset: str, experiment: str) -> Path:
    return features_dir(dataset, experiment) / "feature_importance.csv"


def labels_csv_path(dataset: str, experiment: str) -> Path:
    return features_dir(dataset, experiment) / "labels.csv"


def figures_dir(dataset: str, experiment: str) -> Path:
    return results_dir(dataset, experiment) / "figures"
