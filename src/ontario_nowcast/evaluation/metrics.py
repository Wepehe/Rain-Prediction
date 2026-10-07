"""Meteorological categorical, probabilistic, and continuous metrics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def contingency_counts(observed: np.ndarray, forecast: np.ndarray, threshold: float) -> dict[str, int]:
    valid = np.isfinite(observed) & np.isfinite(forecast)
    obs = observed[valid] >= threshold
    pred = forecast[valid] >= threshold
    return {
        "hits": int(np.sum(obs & pred)),
        "misses": int(np.sum(obs & ~pred)),
        "false_alarms": int(np.sum(~obs & pred)),
        "correct_negatives": int(np.sum(~obs & ~pred)),
    }


def categorical_metrics(observed: np.ndarray, forecast: np.ndarray, threshold: float) -> dict[str, float]:
    counts = contingency_counts(observed, forecast, threshold)
    h, m, f, c = (counts[k] for k in ("hits", "misses", "false_alarms", "correct_negatives"))

    def ratio(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")

    total = h + m + f + c
    random_hits = ((h + m) * (h + f) / total) if total else 0.0
    return {
        **counts,
        "csi": ratio(h, h + m + f),
        "pod": ratio(h, h + m),
        "far": ratio(f, h + f),
        "precision": ratio(h, h + f),
        "recall": ratio(h, h + m),
        "f1": ratio(2 * h, 2 * h + f + m),
        "ets": ratio(h - random_hits, h + m + f - random_hits),
    }


def brier_score(observed_occurrence: np.ndarray, probability: np.ndarray) -> float:
    valid = np.isfinite(observed_occurrence) & np.isfinite(probability)
    if not np.any(valid):
        return float("nan")
    probs = probability[valid]
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return float(np.mean((probs - observed_occurrence[valid]) ** 2))


def fractions_skill_score(
    observed: np.ndarray, forecast: np.ndarray, threshold: float, neighbourhood: int
) -> float:
    if neighbourhood < 1 or neighbourhood % 2 == 0:
        raise ValueError("neighbourhood must be a positive odd integer")
    valid = np.isfinite(observed) & np.isfinite(forecast)
    obs = uniform_filter(((observed >= threshold) & valid).astype(float), neighbourhood)
    pred = uniform_filter(((forecast >= threshold) & valid).astype(float), neighbourhood)
    denominator = np.sum(obs**2 + pred**2)
    return float(1 - np.sum((obs - pred) ** 2) / denominator) if denominator else 1.0


def neighbourhood_size_from_radius_km(radius_km: float, resolution_km: float) -> int:
    """Convert a physical FSS radius to an odd square neighbourhood size in pixels."""
    if radius_km <= 0:
        raise ValueError("radius_km must be positive")
    if resolution_km <= 0:
        raise ValueError("resolution_km must be positive")
    radius_pixels = int(np.ceil(radius_km / resolution_km))
    return radius_pixels * 2 + 1


def fractions_skill_score_km(
    observed: np.ndarray,
    forecast: np.ndarray,
    threshold: float,
    radius_km: float,
    resolution_km: float,
) -> float:
    """Compute FSS using a physically named neighbourhood radius."""
    neighbourhood = neighbourhood_size_from_radius_km(radius_km, resolution_km)
    return fractions_skill_score(observed, forecast, threshold, neighbourhood)
