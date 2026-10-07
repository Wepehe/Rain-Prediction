import numpy as np
import pandas as pd

from ontario_nowcast.events.geometry import polygon_mask
from ontario_nowcast.events.tracking import build_independent_initiation_catalog


def test_polygon_mask_keeps_points_inside_simple_polygon() -> None:
    lon, lat = np.meshgrid(np.arange(4), np.arange(4))
    mask = polygon_mask(lon, lat, ((0, 0), (3, 0), (3, 3), (0, 3), (0, 0)))
    assert bool(mask[1, 1])
    assert not bool(mask[3, 3])


def test_tracking_links_duplicate_initiation_components() -> None:
    rates = np.zeros((28, 12, 12), dtype=np.float32)
    rates[5:8, 5:8, 5:8] = 3.0
    times = pd.date_range("2024-07-16T12:00:00Z", periods=len(rates), freq="6min")
    latitude = np.linspace(42.0, 44.0, 12)
    longitude = np.linspace(-82.0, -78.0, 12)
    catalog = build_independent_initiation_catalog(
        rates,
        times,
        latitude,
        longitude,
        interval_minutes=6,
        resolution_km=2.0,
        history_minutes=30,
        horizon_minutes=60,
        minimum_pixels=4,
        opening_pixels=0,
        closing_pixels=0,
        link_max_gap_minutes=18,
        link_max_centroid_distance_km=20.0,
        polygon=((-83, 41), (-77, 41), (-77, 45), (-83, 45), (-83, 41)),
    )
    assert len(catalog) == 1
    assert catalog.iloc[0]["component_count"] >= 1
    assert bool(catalog.iloc[0]["apparent_initiation"])
