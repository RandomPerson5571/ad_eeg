#!/usr/bin/env python3
"""Print dataset citation and validate local EEG_data layout."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS, RAW_DATA_DIR

CITATION = """
Dataset: Miltiadous et al. (2023)
Title: A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG
DOI: https://doi.org/10.3390/data8060095

Place downloaded data under EEG_data/ with this layout per dataset:

  EEG_data/dataset2/
    participants.tsv
    sub-001/eeg/sub-001_task-eyesclosed_eeg.set
    sub-002/eeg/...
    derivatives/sub-001/eeg/...  (optional, for reference preprocessing)

Repeat for dataset3 if using both datasets (DATASETS in config.py).
"""


def validate_dataset(dataset_id):
    root = Path(RAW_DATA_DIR) / f"dataset{dataset_id}"
    issues = []

    participants = root / "participants.tsv"
    if not participants.exists():
        issues.append(f"Missing {participants}")

    sub_dirs = sorted(root.glob("sub-*/eeg/*_task-eyesclosed_eeg.set"))
    if not sub_dirs:
        issues.append(f"No .set files found under {root}/sub-*/eeg/")

    return issues, len(sub_dirs)


def main():
    print(CITATION)

    all_ok = True
    for dataset_id in DATASETS:
        issues, n_files = validate_dataset(dataset_id)
        if issues:
            all_ok = False
            print(f"\ndataset{dataset_id}: INVALID")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\ndataset{dataset_id}: OK ({n_files} .set files found)")

    if not all_ok:
        print("\nDownload the dataset from the DOI above and extract into EEG_data/.")
        sys.exit(1)

    print("\nData layout validated. Run: python scripts/ingest_features.py --all-datasets --all")


if __name__ == "__main__":
    main()
