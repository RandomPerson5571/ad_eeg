"""Render .fif checkpoints in tests/data for visual inspection."""

from pathlib import Path

import mne

DATA_DIR = Path(__file__).resolve().parents[1] / "eeg_visual_test" / "data"


def load_fif(path: Path) -> mne.io.BaseRaw | mne.Epochs:
    if path.stem.endswith("_epo"):
        return mne.read_epochs(path, preload=True)
    return mne.io.read_raw_fif(path, preload=True)


def main() -> None:
    if not DATA_DIR.is_dir():
        print(f"Directory not found: {DATA_DIR}")
        return

    fif_files = sorted(DATA_DIR.glob("*.fif"))
    if not fif_files:
        print(f"No .fif files in {DATA_DIR}")
        return

    for path in fif_files:
        print(f"\n--- {path.name} ---")
        obj = load_fif(path)
        print(obj)
        obj.plot(block=True)


if __name__ == "__main__":
    main()
