"""Small geometry helpers without a heavyweight GIS dependency."""

from __future__ import annotations

import numpy as np

SOUTHERN_ONTARIO_POLYGON_WGS84 = (
    (-84.9, 41.6),
    (-82.0, 41.4),
    (-78.8, 43.0),
    (-74.7, 44.4),
    (-76.2, 46.6),
    (-80.0, 47.3),
    (-84.8, 46.5),
    (-84.9, 41.6),
)


def polygon_mask(
    longitude: np.ndarray,
    latitude: np.ndarray,
    polygon: list[list[float]] | tuple[tuple[float, float], ...] = SOUTHERN_ONTARIO_POLYGON_WGS84,
) -> np.ndarray:
    """Return True for grid points inside a lon/lat polygon using ray casting."""
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    if lon.shape != lat.shape:
        raise ValueError("longitude and latitude arrays must have the same shape")
    vertices = np.asarray(polygon, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 4:
        raise ValueError("polygon must contain lon/lat coordinate pairs")
    x = lon.ravel()
    y = lat.ravel()
    inside = np.zeros_like(x, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        denominator = y1 - y0
        denominator = denominator if abs(denominator) > 1e-12 else 1e-12
        crosses = ((y0 > y) != (y1 > y)) & (
            x < (x1 - x0) * (y - y0) / denominator + x0
        )
        inside ^= crosses
        x0, y0 = x1, y1
    return inside.reshape(lon.shape)
