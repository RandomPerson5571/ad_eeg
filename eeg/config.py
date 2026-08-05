"""Configuration loading: base YAML configs + experiment overrides."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    id: int
    task: str
    raw_subdir: str

    @property
    def raw_dir(self) -> Path:
        from eeg.paths import raw_data_root

        return raw_data_root() / self.raw_subdir


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base_configs() -> dict[str, Any]:
    return {
        "dataset": _load_yaml(CONFIGS_DIR / "dataset.yaml"),
        "features": _load_yaml(CONFIGS_DIR / "features.yaml"),
        "training": _load_yaml(CONFIGS_DIR / "training.yaml"),
    }


def _build_alias_map() -> dict[str, str]:
    base = load_base_configs()
    alias_map: dict[str, str] = {"all": "all"}
    for name, spec in base["dataset"]["datasets"].items():
        alias_map[name] = name
        for alias in spec.get("aliases", []):
            alias_map[str(alias)] = name
    return alias_map


def resolve_dataset(tag: str) -> list[DatasetSpec]:
    """Resolve a dataset tag to one or more DatasetSpec objects."""
    alias_map = _build_alias_map()
    key = str(tag).strip().lower()
    if key not in alias_map:
        known = sorted(alias_map.keys())
        raise ValueError(f"Unknown dataset '{tag}'. Known: {', '.join(known)}")

    canonical = alias_map[key]
    base = load_base_configs()
    datasets_cfg = base["dataset"]["datasets"]

    if canonical == "all":
        return [
            DatasetSpec(
                name=name,
                id=spec["id"],
                task=spec["task"],
                raw_subdir=spec["raw_subdir"],
            )
            for name, spec in datasets_cfg.items()
        ]

    spec = datasets_cfg[canonical]
    return [
        DatasetSpec(
            name=canonical,
            id=spec["id"],
            task=spec["task"],
            raw_subdir=spec["raw_subdir"],
        )
    ]


def load_experiment(name: str) -> dict[str, Any]:
    """Load merged config: base configs + experiment overrides."""
    exp_path = EXPERIMENTS_DIR / f"{name}.yaml"
    if not exp_path.exists():
        available = sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Experiment '{name}' not found. Available: {', '.join(available)}"
        )

    merged = load_base_configs()
    experiment = _load_yaml(exp_path)
    merged["experiment"] = experiment

    if "preprocessing" in experiment:
        merged.setdefault("preprocessing", {}).update(experiment["preprocessing"])
    if "epoching" in experiment:
        merged["epoching"] = {**merged.get("epoching", {}), **experiment["epoching"]}
    if "filtering" in experiment:
        merged["filtering"] = {**merged.get("filtering", {}), **experiment["filtering"]}
    if "bad_channels" in experiment:
        merged["bad_channels"] = {**merged.get("bad_channels", {}), **experiment["bad_channels"]}

    return merged


def config_fingerprint(config: dict[str, Any]) -> str:
    """SHA256 of sorted JSON config for idempotency checks."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def experiment_metadata(
    dataset_name: str,
    experiment_name: str,
    config: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    import platform
    import sys
    from datetime import datetime, timezone

    import mne

    prep = config.get("experiment", {}).get("preprocessing", config.get("preprocessing", {}))
    meta = {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "mne_version": mne.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit_hash(),
        "config_fingerprint": config_fingerprint(config),
        "asr_cutoff": prep.get("asr_cutoff"),
        "ica_n_components": prep.get("ica_n_components"),
        "date": datetime.now(timezone.utc).isoformat(),
    }
    meta.update(extra)
    return meta
