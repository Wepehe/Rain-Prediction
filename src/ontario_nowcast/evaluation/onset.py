"""Coordinate-specific onset and cessation timing verification."""

from __future__ import annotations

import numpy as np


def first_crossing_minutes(
    rates: np.ndarray, threshold: float, interval_minutes: int, *, above: bool = True
) -> np.ndarray:
    values = np.asarray(rates, dtype=float)
    if values.ndim < 1:
        raise ValueError("rates must have a lead-time dimension")
    crossed = values >= threshold if above else values < threshold
    any_crossed = np.any(crossed, axis=0)
    first = np.argmax(crossed, axis=0) + 1
    return np.where(any_crossed, first * interval_minutes, np.nan).astype(float)


def onset_metrics(observed_minutes: np.ndarray, predicted_minutes: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed_minutes, dtype=float)
    predicted = np.asarray(predicted_minutes, dtype=float)
    obs_event = np.isfinite(observed)
    pred_event = np.isfinite(predicted)
    paired = obs_event & pred_event
    errors = predicted[paired] - observed[paired]
    result: dict[str, float | int] = {
        "observed_events": int(obs_event.sum()),
        "predicted_events": int(pred_event.sum()),
        "missed_events": int((obs_event & ~pred_event).sum()),
        "false_events": int((~obs_event & pred_event).sum()),
        "paired_events": int(paired.sum()),
    }
    result["median_error_minutes"] = float(np.median(errors)) if errors.size else float("nan")
    result["mae_minutes"] = float(np.mean(np.abs(errors))) if errors.size else float("nan")
    for tolerance in (5, 10, 15, 30):
        result[f"within_{tolerance}_minutes"] = (
            float(np.mean(np.abs(errors) <= tolerance)) if errors.size else float("nan")
        )
    return result

