import numpy as np

from ontario_nowcast.preprocessing.grid import lonlat_to_analysis_grid, transform_coordinates
from ontario_nowcast.preprocessing.units import (
    kelvin_to_celsius,
    kilometres_per_hour_to_metres_per_second,
    wind_components,
)


def test_epsg3978_round_trip_near_toronto() -> None:
    x, y = lonlat_to_analysis_grid(-79.3832, 43.6532)
    longitude, latitude = transform_coordinates(
        x, y, source_crs="EPSG:3978", destination_crs="EPSG:4326"
    )
    assert np.isclose(longitude, -79.3832, atol=1e-7)
    assert np.isclose(latitude, 43.6532, atol=1e-7)


def test_meteorological_unit_conversions() -> None:
    assert np.isclose(kelvin_to_celsius(273.15), 0)
    assert np.isclose(kilometres_per_hour_to_metres_per_second(36), 10)
    u, v = wind_components(np.array([10.0]), np.array([270.0]))
    assert np.isclose(u[0], 10)
    assert np.isclose(v[0], 0, atol=1e-12)

