"""Inference helpers — stub for Phase 1."""

from __future__ import annotations

import time
from typing import Any


def predict_with_timing(pipeline, X) -> tuple[Any, float]:
    """Return predictions and per-sample latency in seconds."""
    t0 = time.perf_counter()
    preds = pipeline.predict(X)
    elapsed = time.perf_counter() - t0
    per_sample = elapsed / max(len(X), 1)
    return preds, per_sample
