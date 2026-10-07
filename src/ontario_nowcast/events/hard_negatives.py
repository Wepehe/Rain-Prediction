"""Mine favourable-environment dry cases without treating missing radar as no rain."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import label


@dataclass(frozen=True)
class HardNegativeCandidate:
    anchor_time: str
    centre_lat: float
    centre_lon: float
    pixel_count: int
    maximum_cape_j_kg: float
    minimum_cloud_temperature_k: float


def mine_hard_negatives(
    radar_rate: np.ndarray,
    cape: np.ndarray,
    cloud_temperature: np.ndarray,
    times: pd.DatetimeIndex,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    history_steps: int = 2,
    horizon_steps: int = 2,
    rain_threshold: float = 0.1,
    cape_threshold: float = 500.0,
    cloud_temperature_threshold: float = 260.0,
    minimum_pixels: int = 9,
) -> pd.DataFrame:
    """Find dry/dry components under a deliberately simple favourable-environment definition."""
    if not (radar_rate.shape == cape.shape == cloud_temperature.shape):
        raise ValueError("radar, CAPE, and cloud-temperature tensors must have the same shape")
    candidates = []
    structure = np.ones((3, 3), dtype=int)
    for anchor in range(history_steps - 1, len(times) - horizon_steps):
        history = radar_rate[anchor - history_steps + 1 : anchor + 1]
        future = radar_rate[anchor + 1 : anchor + horizon_steps + 1]
        fully_observed = np.all(np.isfinite(history), axis=0) & np.all(np.isfinite(future), axis=0)
        dry = np.all(history < rain_threshold, axis=0) & np.all(future < rain_threshold, axis=0)
        favourable = (
            (cape[anchor] >= cape_threshold)
            & (cloud_temperature[anchor] <= cloud_temperature_threshold)
            & np.isfinite(cape[anchor])
            & np.isfinite(cloud_temperature[anchor])
        )
        components, count = label(fully_observed & dry & favourable, structure=structure)
        for component in range(1, count + 1):
            ys, xs = np.where(components == component)
            if len(ys) < minimum_pixels:
                continue
            candidates.append(
                HardNegativeCandidate(
                    anchor_time=times[anchor].isoformat(),
                    centre_lat=float(np.mean(latitude[ys, xs])),
                    centre_lon=float(np.mean(longitude[ys, xs])),
                    pixel_count=len(ys),
                    maximum_cape_j_kg=float(np.max(cape[anchor, ys, xs])),
                    minimum_cloud_temperature_k=float(
                        np.min(cloud_temperature[anchor, ys, xs])
                    ),
                )
            )
    return pd.DataFrame(asdict(candidate) for candidate in candidates)


def mine_fused_file(source: str, destination: str) -> pd.DataFrame:
    payload = np.load(source)
    candidates = mine_hard_negatives(
        payload["radar_rate_mm_hr"],
        payload["nwp_cape_j_kg"],
        payload["goes_c13_k"],
        pd.to_datetime(payload["times"], utc=True),
        payload["latitude"],
        payload["longitude"],
    )
    candidates.to_csv(destination, index=False)
    return candidates

