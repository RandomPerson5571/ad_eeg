#!/usr/bin/env python3
"""Download derived artifacts from Zenodo into the project."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, dest):
    try:
        import urllib.request
    except ImportError:
        raise RuntimeError("urllib required for downloads")

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def fetch_from_zenodo(record_id, manifest_path=MANIFEST_PATH):
    try:
        import urllib.request

        api_url = f"https://zenodo.org/api/records/{record_id}"
        with urllib.request.urlopen(api_url) as resp:
            record = json.loads(resp.read().decode())
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch Zenodo record {record_id}: {exc}") from exc

    files = record.get("files", [])
    if not files:
        raise RuntimeError(f"No files in Zenodo record {record_id}")

    for finfo in files:
        filename = finfo["key"]
        url = finfo["links"]["self"]
        dest = PROJECT_ROOT / filename
        download_file(url, dest)

        expected = finfo.get("checksum")
        if expected and expected.startswith("md5:"):
            # ponytail: skip md5 verify unless manifest pins sha256
            pass

    manifest = {
        "zenodo_record_id": record_id,
        "doi": record.get("doi", ""),
        "files": [f["key"] for f in files],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest updated at {manifest_path}")


def fetch_from_manifest(manifest_path=MANIFEST_PATH):
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Upload artifacts to Zenodo first, "
            "then update data/manifest.json or pass --record-id."
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    record_id = manifest.get("zenodo_record_id")
    if record_id:
        fetch_from_zenodo(record_id, manifest_path)
    else:
        print("Manifest has no zenodo_record_id. Update data/manifest.json after Zenodo upload.")


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch derived artifacts from Zenodo.")
    parser.add_argument("--record-id", type=str, help="Zenodo record ID to download.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.record_id:
        fetch_from_zenodo(args.record_id)
    else:
        fetch_from_manifest()
