import numpy as np

from ontario_nowcast.training.stage7_diagnose import _spatial


def test_spatial_diagnostic_uses_two_km_grid():
    observed = np.zeros((5, 5), dtype=bool)
    hrrr = np.zeros((5, 5), dtype=bool)
    observed[2, 2] = True
    hrrr[2, 4] = True
    result = _spatial(observed, hrrr)
    assert result["minimum_distance_km"] == 4.0
    assert result["hrrr_spatial_class"] == "near miss"
