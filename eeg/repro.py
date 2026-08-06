"""Reproducibility helpers: seed pinning and environment snapshots."""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

from eeg.config import config_fingerprint, git_commit_hash


def init_repro(seed: int = 42) -> dict[str, int]:
    """Pin random seeds for stdlib, numpy, and sklearn."""
    random.seed(seed)
    np.random.seed(seed)
    return {"seed": seed}


def _package_version(name: str) -> str | None:
    try:
        import importlib.metadata as meta

        return meta.version(name)
    except Exception:
        return None


def snapshot_environment() -> dict[str, Any]:
    """Capture runtime versions, git SHA, and optional CUDA info."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_version": sys.version.split()[0],
        "numpy": _package_version("numpy"),
        "sklearn": _package_version("scikit-learn"),
        "xgboost": _package_version("xgboost"),
        "mne": _package_version("mne"),
        "mne_version": _package_version("mne"),
        "autoreject": _package_version("autoreject"),
        "autoreject_version": _package_version("autoreject"),
        "asrpy": _package_version("asrpy"),
        "asrpy_version": _package_version("asrpy"),
        "pandas": _package_version("pandas"),
        "git_sha": git_commit_hash() or "unknown",
        "git_commit": git_commit_hash() or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch"] = None
        env["cuda_available"] = False
    return env


def preprocessing_fingerprint(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Canonical preprocessing fingerprint for dataset reports."""
    env = snapshot_environment()
    fp: dict[str, Any] = {
        "mne_version": env.get("mne_version"),
        "autoreject_version": env.get("autoreject_version"),
        "asrpy_version": env.get("asrpy_version"),
        "git_commit": env.get("git_commit"),
        "python_version": env.get("python_version"),
        "timestamp": env.get("timestamp"),
    }
    if config is not None:
        fp["config_sha256"] = config_fingerprint(config)
    return fp


def attach_repro_metadata(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Embed reproducibility fields into a result/metadata dict."""
    env = snapshot_environment()
    result["config"] = config
    result["experiment"] = config.get("experiment", config.get("experiment_name"))
    result["seed"] = config.get("seed", 42)
    result["timestamp"] = env["timestamp"]
    result["git_sha"] = env["git_sha"]
    result["environment"] = env
    return result
