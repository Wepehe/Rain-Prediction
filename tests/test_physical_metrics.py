import numpy as np
import pytest

from ontario_nowcast.evaluation.metrics import (
    fractions_skill_score_km,
    neighbourhood_size_from_radius_km,
)


def test_neighbourhood_size_from_radius_km_is_odd_diameter() -> None:
    assert neighbourhood_size_from_radius_km(6.0, 2.0) == 7
    assert neighbourhood_size_from_radius_km(6.1, 2.0) == 9


def test_fss_km_matches_perfect_forecast() -> None:
    observed = np.zeros((8, 8))
    observed[3:5, 3:5] = 1.0
    assert fractions_skill_score_km(observed, observed, 0.1, 6.0, 2.0) == pytest.approx(1.0)
