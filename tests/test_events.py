import numpy as np

from ontario_nowcast.events.detect import detect_event_masks


def test_clear_to_rain_and_missing_are_distinct() -> None:
    rates = np.zeros((8, 2, 2), dtype=float)
    rates[4:, 0, 0] = 2.0
    rates[1, 1, 1] = np.nan
    events = detect_event_masks(rates, history_steps=3, horizon_steps=3, max_missing_fraction=0)
    assert events.clear_to_rain[2, 0, 0]
    assert not events.valid[2, 1, 1]
    assert not events.clear_to_rain[2, 1, 1]


def test_event_split_anchor_does_not_peek_beyond_horizon() -> None:
    rates = np.zeros((10, 1, 1), dtype=float)
    rates[9, 0, 0] = 5
    events = detect_event_masks(rates, history_steps=2, horizon_steps=2)
    assert not events.clear_to_rain[:7].any()

