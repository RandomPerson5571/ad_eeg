import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from config import DATASETS, FEATURE_COLUMNS, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from util.io import load_features_df


def validate_feature_schema(df):
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if not missing:
        return

    stale = {"lzc", "mse_mean"} & set(df.columns)
    hint = " Re-ingest required to refresh feature columns." if stale else ""
    raise ValueError(f"Feature parquet missing columns: {missing}.{hint}")


def subject_level_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    group_col="participant_id",
    label_col="label",
):
    """Split at subject level so epochs from one participant stay in one fold."""
    subjects = df[[group_col, label_col]].drop_duplicates()
    groups = subjects[group_col].values
    labels = subjects[label_col].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(groups, labels, groups=groups))

    train_subjects = set(subjects.iloc[train_idx][group_col])
    test_subjects = set(subjects.iloc[test_idx][group_col])

    train_mask = df[group_col].isin(train_subjects)
    test_mask = df[group_col].isin(test_subjects)

    x_train = df.loc[train_mask, FEATURE_COLUMNS]
    x_test = df.loc[test_mask, FEATURE_COLUMNS]
    y_train = df.loc[train_mask, label_col]
    y_test = df.loc[test_mask, label_col]

    split_ids = {
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "test_size": test_size,
        "random_state": random_state,
    }

    return x_train, x_test, y_train, y_test, split_ids


def prepare_training_data(dataset_id, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    if dataset_id not in DATASETS:
        raise ValueError(f"dataset_id must be one of {DATASETS}, got {dataset_id}")

    df = load_features_df(dataset_id=dataset_id)
    validate_feature_schema(df)
    x_train, x_test, y_train, y_test, split_ids = subject_level_split(
        df, test_size=test_size, random_state=random_state
    )
    split_ids["dataset_id"] = dataset_id
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    return x_train, x_test, y_train_enc, y_test_enc, label_encoder, split_ids


def results_path_for_dataset(filename, dataset_id):
    stem, ext = os.path.splitext(filename)
    return os.path.join(RESULTS_DIR, f"{stem}_dataset{dataset_id}{ext}")


def save_metrics(metrics: dict[str, Any], path=None):
    path = path or os.path.join(RESULTS_DIR, "metrics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


def save_model_artifact(
    pipeline,
    label_encoder,
    feature_names,
    split_ids,
    model_name,
    output_path,
    extra=None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "feature_names": list(feature_names),
        "split_ids": split_ids,
        "model_name": model_name,
    }
    if extra:
        artifact.update(extra)
    joblib.dump(artifact, output_path)
    print(f"Saved model artifact to {output_path}")
    return artifact


def classification_metrics(y_true, y_pred, label_encoder):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
