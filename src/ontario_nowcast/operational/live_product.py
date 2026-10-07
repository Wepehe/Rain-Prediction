"""Consumer summaries for live Residual V1 output; no forecast logic lives here."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from .inference import OCCURRENCE_THRESHOLD, OperationalForecast


def point_from_latlon(metadata: dict[str, Any], latitude: float, longitude: float) -> tuple[int, int]:
    grid = metadata.get("grid", {})
    lat = np.asarray(grid.get("latitude"))
    lon = np.asarray(grid.get("longitude"))
    if lat.shape != (128, 128) or lon.shape != (128, 128):
        raise ValueError("forecast metadata lacks the 128x128 geographic grid")
    distance = (lat - latitude) ** 2 + ((lon - longitude) * np.cos(np.deg2rad(latitude))) ** 2
    y, x = np.unravel_index(np.nanargmin(distance), distance.shape)
    return int(y), int(x)


def location_summary(result: OperationalForecast, y: int, x: int) -> dict[str, Any]:
    probability = result.residual_probability[:, y, x]
    rate = result.residual_rate_mm_h[:, y, x]
    wet = probability >= OCCURRENCE_THRESHOLD
    first = int((np.argmax(wet) + 1) * 6) if wet.any() else None
    currently_wet = bool(result.metadata.get("latest_rate_at_location_mm_h", 0) > 0.1)
    if currently_wet and wet.any():
        status = "Rain is present and the model continues to forecast precipitation."
    elif first is not None:
        status = f"Model rain signal begins around +{first} minutes."
    else:
        status = "No operational rain signal is currently forecast at this location."
    return {
        "status": status,
        "next_30_probability_max": float(probability[:5].max()),
        "next_30_rate_max_mm_h": float(rate[:5].max()),
        "next_60_probability_max": float(probability[:10].max()),
        "next_60_rate_max_mm_h": float(rate[:10].max()),
        "next_120_probability_max": float(probability.max()),
        "next_120_rate_max_mm_h": float(rate.max()),
        "rain_likely": bool(wet.any()),
        "first_wet_lead_minutes": first,
    }


def freshness_minutes(issue_time: datetime, now: datetime) -> float:
    return max(0.0, (now - issue_time).total_seconds() / 60)


def has_newer_issue(current_issue: datetime, objects) -> bool:
    from ..data.sample import _timestamp_from_key
    return bool(objects) and max(_timestamp_from_key(obj.key) for obj in objects) > current_issue
