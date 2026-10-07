import numpy as np
import pandas as pd

from ontario_nowcast.events.hard_negatives import mine_hard_negatives


def test_hard_negative_requires_observed_dry_radar() -> None:
    radar = np.zeros((5, 4, 4))
    cape = np.full_like(radar, 1_000)
    cloud = np.full_like(radar, 240)
    radar[:, 0, 0] = np.nan
    result = mine_hard_negatives(
        radar,
        cape,
        cloud,
        pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
        np.broadcast_to(np.arange(4)[:, None], (4, 4)),
        np.broadcast_to(np.arange(4)[None, :], (4, 4)),
        minimum_pixels=4,
    )
    assert len(result) == 2
    assert result["pixel_count"].eq(15).all()

