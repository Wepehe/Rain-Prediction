import numpy as np

from ontario_nowcast.preprocessing.fuse import _interpolate_regular_latlon, analysis_grid


def test_analysis_grid_is_two_kilometres_and_covers_bbox() -> None:
    x, y, longitude, latitude = analysis_grid((-80.0, 43.0, -79.0, 44.0), 2_000)
    assert np.all(np.diff(x) == 2_000)
    assert np.all(np.diff(y) == 2_000)
    assert longitude.shape == latitude.shape == (len(y), len(x))
    assert longitude.min() < -79.9 and longitude.max() > -79.1
    assert latitude.min() < 43.1 and latitude.max() > 43.9


def test_regular_latlon_interpolation_handles_descending_latitude() -> None:
    latitude = np.array([2.0, 1.0, 0.0])
    longitude = np.array([0.0, 1.0, 2.0])
    values = latitude[:, None] + longitude[None, :]
    result = _interpolate_regular_latlon(
        values, latitude, longitude, np.array([[0.5]]), np.array([[1.5]])
    )
    assert np.isclose(result[0, 0], 2.0)

