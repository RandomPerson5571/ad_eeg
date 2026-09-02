"""Regression tests for resumable per-subject feature extraction."""

import json

import pandas as pd

from scripts import extract_features as extract


def test_subject_done_requires_matching_partition_log_and_epoch_hash(
    tmp_path, monkeypatch
):
    epoch = tmp_path / "sub-001_epo.fif"
    epoch.write_bytes(b"epochs")
    partition = tmp_path / "subject_partitions" / "sub-001.parquet"
    log = tmp_path / "logs" / "sub-001.json"
    extract._write_partition(
        pd.DataFrame({"participant_id": ["sub-001"], "power": [1.0]}),
        partition,
    )
    log.parent.mkdir()
    log.write_text(
        json.dumps(
            {
                "status": "ok",
                "config_fingerprint": "config-a",
                "epochs_sha256": extract.sha256_file(epoch),
                "partition": partition.name,
            }
        )
    )
    monkeypatch.setattr(extract, "_partition_path", lambda *_: partition)
    monkeypatch.setattr(extract, "subject_log_path", lambda *_, **__: log)

    assert extract._subject_done(
        "eyesclosed", "baseline", "sub-001", epoch, "config-a"
    )
    assert not extract._subject_done(
        "eyesclosed", "baseline", "sub-001", epoch, "config-b"
    )


def test_legacy_temporary_partition_is_adopted_as_durable_checkpoint(
    tmp_path, monkeypatch
):
    feature_root = tmp_path / "features"
    legacy = feature_root / ".feature-parts-old" / "sub-001.parquet"
    legacy.parent.mkdir(parents=True)
    pd.DataFrame(
        {"participant_id": ["sub-001", "sub-001"], "power": [1.0, 2.0]}
    ).to_parquet(legacy, index=False)
    epoch = tmp_path / "sub-001_epo.fif"
    epoch.write_bytes(b"epochs")
    durable = feature_root / "subject_partitions" / "sub-001.parquet"
    log = feature_root / "logs" / "sub-001.json"
    monkeypatch.setattr(
        extract, "features_parquet_path", lambda *_: feature_root / "subject_features.parquet"
    )
    monkeypatch.setattr(extract, "_partition_path", lambda *_: durable)
    monkeypatch.setattr(extract, "resolve_checkpoint_path", lambda *_: epoch)
    monkeypatch.setattr(extract, "subject_log_path", lambda *_, **__: log)
    participants = pd.DataFrame(
        [{"participant_id": "sub-001", "Group": "A"}]
    )

    adopted = extract._adopt_legacy_partitions(
        "eyesclosed", "baseline", participants, "config-a"
    )

    assert adopted == 1
    assert pd.read_parquet(durable)["power"].tolist() == [1.0, 2.0]
    saved_log = json.loads(log.read_text())
    assert saved_log["status"] == "ok"
    assert saved_log["partition"] == "sub-001.parquet"
    assert saved_log["recovered_from_legacy_partition"] is True
