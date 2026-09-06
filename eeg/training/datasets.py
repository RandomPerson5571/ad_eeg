"""Data loading and subject-level splits for training."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from eeg.config import load_base_configs, load_experiment
from eeg.io import load_features_df
from eeg.paths import selected_features_path


def feature_columns(
    config: dict | None = None, feature_set: str | None = None
) -> list[str]:
    """Return the configured columns for a named feature set.

    ``load_base_configs()`` stores feature settings under ``features`` while
    notebook run configs are intentionally flat. Supporting both shapes keeps
    the existing local callers compatible and lets the Kaggle orchestrator
    select a feature set without rewriting the feature parquet.
    """
    cfg = config or load_base_configs()
    features_cfg = cfg.get("features", cfg)
    selected_name = feature_set or cfg.get("feature_set")
    feature_sets = features_cfg.get("feature_sets", {})
    if selected_name and feature_sets:
        if selected_name not in feature_sets:
            available = ", ".join(sorted(feature_sets))
            raise ValueError(
                f"Unknown feature set {selected_name!r}; available: {available}"
            )
        selected = feature_sets[selected_name]
        if isinstance(selected, dict):
            selected = selected.get("columns", selected.get("feature_columns"))
        if not selected:
            raise ValueError(f"Feature set {selected_name!r} does not define columns")
        return list(selected)
    return list(features_cfg["feature_columns"])


def validate_feature_schema(df: pd.DataFrame, config: dict | None = None) -> None:
    cols = feature_columns(config)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature parquet missing columns: {missing}")


def load_selected_features(dataset: str, experiment: str) -> pd.DataFrame:
    path = selected_features_path(dataset, experiment)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run feature selection first.")
    return pd.read_parquet(path)


def subject_level_split(
    df: pd.DataFrame,
    config: dict | None = None,
    group_col: str = "participant_id",
    label_col: str = "label",
):
    cfg = config or load_base_configs()
    train_cfg = cfg.get("training", cfg)
    test_size = train_cfg.get("test_size", 0.2)
    random_state = train_cfg.get("random_state", cfg.get("seed", 42))
    cols = feature_columns(cfg)

    subjects = df[[group_col, label_col]].drop_duplicates()
    groups = subjects[group_col].values
    labels = subjects[label_col].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(groups, labels, groups=groups))

    train_subjects = set(subjects.iloc[train_idx][group_col])
    test_subjects = set(subjects.iloc[test_idx][group_col])
    train_mask = df[group_col].isin(train_subjects)
    test_mask = df[group_col].isin(test_subjects)

    split_ids = {
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "test_size": test_size,
        "random_state": random_state,
    }
    return (
        df.loc[train_mask, cols],
        df.loc[test_mask, cols],
        df.loc[train_mask, label_col],
        df.loc[test_mask, label_col],
        split_ids,
    )


def prepare_data(
    dataset_name: str,
    experiment: str,
    dataset_id: int,
    config: dict | None = None,
    feature_cols: list[str] | None = None,
    use_selected: bool = False,
):
    if use_selected or feature_cols is not None:
        try:
            df = load_selected_features(dataset_name, experiment)
        except FileNotFoundError:
            df = load_features_df(dataset_name, experiment, dataset_id)
    else:
        df = load_features_df(dataset_name, experiment, dataset_id)

    cfg = config or load_experiment(experiment)
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing: {missing}")
        cols = feature_cols
    else:
        validate_feature_schema(df, cfg)
        cols = feature_columns(cfg)

    x_train, x_test, y_train, y_test, split_ids = subject_level_split(df, cfg)
    if feature_cols is not None:
        x_train = x_train[cols]
        x_test = x_test[cols]

    from sklearn.preprocessing import LabelEncoder

    split_ids["dataset_name"] = dataset_name
    split_ids["dataset_id"] = dataset_id
    split_ids["experiment"] = experiment
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    return x_train, x_test, y_train_enc, y_test_enc, label_encoder, split_ids
