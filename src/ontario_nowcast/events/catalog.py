"""Turn pixel-level onset masks into a compact, de-duplicated candidate catalogue."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import label


@dataclass(frozen=True)
class InitiationCandidate:
    anchor_time: str
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    centre_lat: float
    centre_lon: float
    pixel_count: int
    future_max_mm_hr: float


def _bbox_iou(a: InitiationCandidate, b: InitiationCandidate) -> float:
    height = max(0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min) + 1)
    width = max(0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min) + 1)
    intersection = height * width
    area_a = (a.y_max - a.y_min + 1) * (a.x_max - a.x_min + 1)
    area_b = (b.y_max - b.y_min + 1) * (b.x_max - b.x_min + 1)
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def build_initiation_catalog(
    masks: np.ndarray,
    rates: np.ndarray,
    times: pd.DatetimeIndex,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    horizon_steps: int,
    minimum_pixels: int = 9,
    temporal_dedup_minutes: int = 30,
) -> pd.DataFrame:
    if masks.shape != rates.shape:
        raise ValueError("masks and rates must have identical (time, y, x) shapes")
    if len(times) != rates.shape[0]:
        raise ValueError("one timestamp is required per frame")
    candidates: list[InitiationCandidate] = []
    structure = np.ones((3, 3), dtype=int)
    for anchor in np.flatnonzero(masks.reshape(len(times), -1).any(axis=1)):
        components, count = label(masks[anchor], structure=structure)
        for component in range(1, count + 1):
            ys, xs = np.where(components == component)
            if len(ys) < minimum_pixels:
                continue
            future = rates[anchor + 1 : min(len(rates), anchor + horizon_steps + 1), ys, xs]
            candidates.append(
                InitiationCandidate(
                    anchor_time=times[anchor].isoformat(),
                    y_min=int(ys.min()),
                    y_max=int(ys.max()),
                    x_min=int(xs.min()),
                    x_max=int(xs.max()),
                    centre_lat=float(np.mean(latitudes[ys])),
                    centre_lon=float(np.mean(longitudes[xs])),
                    pixel_count=len(ys),
                    future_max_mm_hr=float(np.nanmax(future)),
                )
            )

    selected: list[InitiationCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.future_max_mm_hr, reverse=True):
        candidate_time = pd.Timestamp(candidate.anchor_time)
        duplicate = any(
            abs((candidate_time - pd.Timestamp(existing.anchor_time)).total_seconds())
            <= temporal_dedup_minutes * 60
            and _bbox_iou(candidate, existing) >= 0.25
            for existing in selected
        )
        if not duplicate:
            selected.append(candidate)
    return pd.DataFrame(asdict(candidate) for candidate in sorted(selected, key=lambda x: x.anchor_time))
