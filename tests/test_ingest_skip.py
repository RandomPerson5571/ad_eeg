from util.io import merge_ingest_logs


def test_merge_ingest_logs_replaces_same_subject():
    existing = [
        {"dataset_id": 2, "subject_num": 1, "status": "error", "error": "old"},
        {"dataset_id": 2, "subject_num": 2, "status": "ok"},
    ]
    new_entries = [
        {"dataset_id": 2, "subject_num": 1, "status": "ok", "n_epochs": 10},
    ]
    merged = merge_ingest_logs(existing, new_entries)
    by_subject = {e["subject_num"]: e for e in merged}
    assert by_subject[1]["status"] == "ok"
    assert by_subject[1]["n_epochs"] == 10
    assert by_subject[2]["status"] == "ok"
