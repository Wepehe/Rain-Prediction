import numpy as np
import pandas as pd

from ontario_nowcast.events.catalog import build_initiation_catalog


def test_catalog_deduplicates_nearby_temporal_components() -> None:
    rates = np.zeros((8, 10, 10))
    masks = np.zeros_like(rates, dtype=bool)
    masks[2, 3:6, 3:6] = True
    masks[3, 3:6, 3:6] = True
    rates[4, 3:6, 3:6] = 10
    times = pd.date_range("2024-01-01", periods=8, freq="6min", tz="UTC")
    catalog = build_initiation_catalog(
        masks, rates, times, np.arange(10), np.arange(10), horizon_steps=3, minimum_pixels=9
    )
    assert len(catalog) == 1
    assert catalog.iloc[0].future_max_mm_hr == 10

