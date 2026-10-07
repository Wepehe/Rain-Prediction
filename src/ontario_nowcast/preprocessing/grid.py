"""Explicit coordinate transforms for the Canadian analysis grid."""

from __future__ import annotations

import numpy as np


def transform_coordinates(
    x: np.ndarray | float,
    y: np.ndarray | float,
    *,
    source_crs: str,
    destination_crs: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'data' extra for coordinate transforms") from exc
    transformer = Transformer.from_crs(source_crs, destination_crs, always_xy=True)
    transformed_x, transformed_y = transformer.transform(x, y)
    return np.asarray(transformed_x), np.asarray(transformed_y)


def lonlat_to_analysis_grid(
    longitude: np.ndarray | float, latitude: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    return transform_coordinates(
        longitude, latitude, source_crs="EPSG:4326", destination_crs="EPSG:3978"
    )

