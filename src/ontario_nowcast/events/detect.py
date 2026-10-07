"""Vectorized pixel/tile event mining for onset, intensification, and dissipation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventMasks:
    clear_to_rain: np.ndarray
    weak_to_strong: np.ndarray
    dissipation: np.ndarray
    persistent: np.ndarray
    valid: np.ndarray


def detect_event_masks(
    rates: np.ndarray,
    *,
    history_steps: int,
    horizon_steps: int,
    rain_threshold: float = 0.1,
    strong_threshold: float = 5.0,
    max_missing_fraction: float = 0.1,
) -> EventMasks:
    """Classify each anchor pixel from past and future precipitation-rate windows.

    `rates` has shape (time, y, x). NaNs are missing, not dry. An anchor is valid only when both
    windows satisfy the configured missing-data limit.
    """
    values = np.asarray(rates, dtype=float)
    if values.ndim != 3:
        raise ValueError("rates must have shape (time, y, x)")
    if history_steps < 1 or horizon_steps < 1:
        raise ValueError("history_steps and horizon_steps must be positive")
    n_time, ny, nx = values.shape
    shape = (n_time, ny, nx)
    output = {name: np.zeros(shape, dtype=bool) for name in ("clear", "strong", "diss", "persist", "valid")}

    for anchor in range(history_steps - 1, n_time - horizon_steps):
        history = values[anchor - history_steps + 1 : anchor + 1]
        future = values[anchor + 1 : anchor + horizon_steps + 1]
        missing = np.concatenate((np.isnan(history), np.isnan(future))).mean(axis=0)
        valid = missing <= max_missing_fraction
        past_max = np.nanmax(history, axis=0)
        future_max = np.nanmax(future, axis=0)
        future_min = np.nanmin(future, axis=0)
        now = values[anchor]
        output["valid"][anchor] = valid
        output["clear"][anchor] = valid & (past_max < rain_threshold) & (future_max >= rain_threshold)
        output["strong"][anchor] = valid & (now >= rain_threshold) & (now < strong_threshold) & (future_max >= strong_threshold)
        output["diss"][anchor] = valid & (now >= rain_threshold) & (future_min < rain_threshold)
        output["persist"][anchor] = valid & (np.nanmin(history, axis=0) >= rain_threshold) & (future_min >= rain_threshold)

    return EventMasks(
        clear_to_rain=output["clear"],
        weak_to_strong=output["strong"],
        dissipation=output["diss"],
        persistent=output["persist"],
        valid=output["valid"],
    )

