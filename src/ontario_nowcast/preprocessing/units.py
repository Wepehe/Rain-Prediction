"""Small, tested meteorological unit conversions."""

from __future__ import annotations

import numpy as np


def kelvin_to_celsius(values: np.ndarray | float) -> np.ndarray:
    return np.asarray(values, dtype=float) - 273.15


def kilometres_per_hour_to_metres_per_second(values: np.ndarray | float) -> np.ndarray:
    return np.asarray(values, dtype=float) / 3.6


def wind_components(speed: np.ndarray, direction_from_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert meteorological wind-from direction and speed to eastward/northward components."""
    speed = np.asarray(speed, dtype=float)
    radians = np.deg2rad(np.asarray(direction_from_degrees, dtype=float))
    return -speed * np.sin(radians), -speed * np.cos(radians)

