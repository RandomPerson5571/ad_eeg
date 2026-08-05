"""Reproducibility helpers: seed pinning and environment snapshots."""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

from eeg.config import git_commit_hash


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
        "numpy": _package_version("numpy"),
        "sklearn": _package_version("scikit-learn"),
        "xgboost": _package_version("xgboost"),
        "mne": _package_version("mne"),
        "pandas": _package_version("pandas"),
        "git_sha": git_commit_hash() or "unknown",
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
