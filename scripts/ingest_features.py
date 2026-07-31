#!/usr/bin/env python3
"""Batch EEG ingestion: preprocess raw .set files and extract features to parquet."""

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

from config import DATASETS, FAST_PREPROCESS_DEFAULTS, PREPROCESS_DEFAULTS, SAMPLING_RATE  # noqa: E402
from util.extract_features import connectivity_for_epochs, extract_eeg_features  # noqa: E402
from util.io import (  # noqa: E402
    list_subjects,
    merge_ingest_logs,
    raw_eeg_path,
    read_eeg_data,
    read_ingest_log,
    save_as_parquet,
    save_clean_eeg,
    write_ingest_log,
)
from util.preprocessing import preprocess_EEG  # noqa: E402


def process_subject(dataset_id, subject_num, participant_df, save_clean=False, qc=False, fast=False):
    row = participant_df.iloc[subject_num - 1]
    participant_id = row["participant_id"]
    label = row["Group"]

    path = raw_eeg_path(dataset_id, subject_num)
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing EEG file: {path}")

    preprocess_kwargs = FAST_PREPROCESS_DEFAULTS if fast else PREPROCESS_DEFAULTS

    t0 = time.perf_counter()
    eeg_raw = read_eeg_data(path, sfreq=SAMPLING_RATE)
    ch_names = eeg_raw.ch_names

    clean_eeg, epochs = preprocess_EEG(eeg_raw, **preprocess_kwargs)

    if save_clean:
        save_clean_eeg(clean_eeg, dataset_id, subject_num)

    if qc:
        import matplotlib.pyplot as plt

        psd = clean_eeg.compute_psd(method="welch", fmin=1, fmax=40)
        psd.plot()
        plt.savefig(PROJECT_ROOT / "results" / f"qc_psd_dataset{dataset_id}_sub{subject_num:03d}.png")
        plt.close()

    data = epochs.get_data()
    subject_connectivity = connectivity_for_epochs(epochs, ch_names)
    all_epoch_features = extract_eeg_features(data, ch_names=ch_names, subject_connectivity=subject_connectivity)
    save_as_parquet(all_epoch_features, participant_id, dataset_id, label)

    elapsed = time.perf_counter() - t0
    return {
        "dataset_id": dataset_id,
        "subject_num": subject_num,
        "participant_id": participant_id,
        "label": label,
        "n_epochs": len(all_epoch_features),
        "elapsed_seconds": round(elapsed, 2),
        "preprocessing_mode": "fast" if fast else "full",
        "status": "ok",
    }


def run_ingest(datasets, subject_nums, limit=None, save_clean=False, qc=False, fast=False, skip_existing=True):
    existing_log = read_ingest_log()
    done = {
        (e["dataset_id"], e["subject_num"])
        for e in existing_log
        if e.get("status") == "ok"
    }
    log = []
    skipped = 0
    participant_tables = {d: list_subjects(d) for d in datasets}

    for dataset_id in datasets:
        nums = subject_nums or list(range(1, len(participant_tables[dataset_id]) + 1))
        if limit:
            nums = nums[:limit]

        for subject_num in nums:
            if skip_existing and (dataset_id, subject_num) in done:
                skipped += 1
                print(
                    f"SKIP: dataset {dataset_id} subject {subject_num:03d} (already ingested)",
                    flush=True,
                )
                continue

            print(f"Processing dataset {dataset_id} subject {subject_num:03d}", flush=True)
            try:
                entry = process_subject(
                    dataset_id,
                    subject_num,
                    participant_tables[dataset_id],
                    save_clean=save_clean,
                    qc=qc,
                    fast=fast,
                )
                log.append(entry)
                print(f"  OK: {entry['n_epochs']} epochs in {entry['elapsed_seconds']}s", flush=True)
            except Exception as exc:
                log.append(
                    {
                        "dataset_id": dataset_id,
                        "subject_num": subject_num,
                        "status": "error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"  ERROR: {exc}", flush=True)

    merged_log = merge_ingest_logs(existing_log, log)
    write_ingest_log(merged_log)
    if log or skipped:
        from scripts.run_pipeline import save_preprocessing_config

        save_preprocessing_config()
    ok_this_run = sum(1 for e in log if e.get("status") == "ok")
    total_ok = sum(1 for e in merged_log if e.get("status") == "ok")
    print(
        f"Ingest complete: {ok_this_run} processed, {skipped} skipped, {total_ok} total OK",
        flush=True,
    )
    return merged_log


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest raw EEG and extract features to parquet.")
    parser.add_argument("--dataset", type=int, action="append", help="Dataset ID (repeatable).")
    parser.add_argument("--all-datasets", action="store_true", help="Process all datasets in config.")
    parser.add_argument("--subject", type=int, help="Single subject number (1-based).")
    parser.add_argument("--all", action="store_true", help="Process all subjects in selected datasets.")
    parser.add_argument("--limit", type=int, help="Max subjects per dataset (for testing).")
    parser.add_argument("--save-clean", action="store_true", help="Save cleaned EEG to EEG_clean_data/.")
    parser.add_argument("--qc", action="store_true", help="Save PSD QC plots to results/.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Minimal preprocessing (bandpass + epoch + AR only; skips notch, ASR, ICA).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip subjects already marked OK in results/ingest_log.json (default: on).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-process all subjects (replaces parquet rows per participant_id).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.all_datasets:
        datasets = DATASETS
    elif args.dataset:
        datasets = args.dataset
    else:
        datasets = DATASETS

    if args.subject:
        subject_nums = [args.subject]
    elif args.all:
        subject_nums = None
    else:
        subject_nums = [1]

    run_ingest(
        datasets=datasets,
        subject_nums=subject_nums,
        limit=args.limit,
        save_clean=args.save_clean,
        qc=args.qc,
        fast=args.fast,
        skip_existing=args.skip_existing,
    )
