"""Timestamp alignment that preserves missingness explicitly."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def align_nearest(
    source_times: Sequence[pd.Timestamp],
    target_times: Sequence[pd.Timestamp],
    tolerance: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray]:
    """Map targets to nearest source indices; use -1 and False when outside tolerance."""
    source = pd.DatetimeIndex(source_times)
    target = pd.DatetimeIndex(target_times)
    if source.tz is None or target.tz is None:
        raise ValueError("All timestamps must be timezone-aware")
    if not source.is_monotonic_increasing or source.has_duplicates:
        raise ValueError("source_times must be strictly increasing")
    indices = source.get_indexer(target, method="nearest", tolerance=tolerance)
    valid = indices >= 0
    return indices.astype(np.int64), valid


def assert_regular_without_filling(
    times: Sequence[pd.Timestamp], expected_interval: pd.Timedelta
) -> np.ndarray:
    """Return a validity mask on a regular timeline; never manufacture observations."""
    index = pd.DatetimeIndex(times)
    if index.empty:
        return np.array([], dtype=bool)
    if index.tz is None:
        raise ValueError("timestamps must be timezone-aware")
    expected = pd.date_range(index.min(), index.max(), freq=expected_interval)
    return expected.isin(index)

