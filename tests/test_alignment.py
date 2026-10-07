import pandas as pd
import pytest

from ontario_nowcast.preprocessing.align import align_nearest, assert_regular_without_filling


def test_nearest_alignment_exposes_missing_values() -> None:
    source = pd.to_datetime(["2024-07-16T12:00Z", "2024-07-16T12:12Z"])
    target = pd.to_datetime(["2024-07-16T12:01Z", "2024-07-16T12:06Z", "2024-07-16T12:11Z"])
    indices, valid = align_nearest(source, target, pd.Timedelta("2min"))
    assert indices.tolist() == [0, -1, 1]
    assert valid.tolist() == [True, False, True]


def test_alignment_requires_timezone() -> None:
    with pytest.raises(ValueError):
        align_nearest(pd.to_datetime(["2024-01-01"]), pd.to_datetime(["2024-01-01"]), pd.Timedelta("1h"))


def test_regular_timeline_does_not_fill_gap() -> None:
    times = pd.to_datetime(["2024-07-16T12:00Z", "2024-07-16T12:12Z"])
    assert assert_regular_without_filling(times, pd.Timedelta("6min")).tolist() == [True, False, True]

