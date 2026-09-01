"""Regression tests for safe batch feature aggregation."""

import pandas as pd

from eeg.io import merge_features_parquet, prepare_feature_rows


def test_merge_features_parquet_keeps_every_subject_and_replaces_updates(
    tmp_path, monkeypatch
):
    output = tmp_path / "subject_features.parquet"
    monkeypatch.setattr(
        "eeg.io.features_parquet_path",
        lambda dataset, experiment: output,
    )

    old = prepare_feature_rows(
        pd.DataFrame({"power": [1.0]}),
        "sub-001",
        "eyesclosed",
        "A",
        2,
    )
    merge_features_parquet([old], "eyesclosed", "baseline")

    updated = prepare_feature_rows(
        pd.DataFrame({"power": [10.0, 11.0]}),
        "sub-001",
        "eyesclosed",
        "A",
        2,
    )
    second = prepare_feature_rows(
        pd.DataFrame({"power": [20.0]}),
        "sub-002",
        "eyesclosed",
        "F",
        2,
    )
    merge_features_parquet([updated, second], "eyesclosed", "baseline")

    result = pd.read_parquet(output).sort_values(["participant_id", "power"])
    assert result["participant_id"].tolist() == ["sub-001", "sub-001", "sub-002"]
    assert result["power"].tolist() == [10.0, 11.0, 20.0]
    assert not list(tmp_path.glob(".subject_features.*.parquet"))
