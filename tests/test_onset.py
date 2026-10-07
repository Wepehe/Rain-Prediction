import numpy as np

from ontario_nowcast.evaluation.onset import first_crossing_minutes, onset_metrics


def test_onset_lead_starts_at_one_interval() -> None:
    rates = np.array([[0, 0], [1, 0], [2, 1]], dtype=float)
    onset = first_crossing_minutes(rates, 0.1, 5)
    assert onset.tolist() == [10.0, 15.0]


def test_onset_metrics_count_misses_and_false_events() -> None:
    observed = np.array([10, np.nan, 30, 20])
    predicted = np.array([15, 5, np.nan, 10])
    metrics = onset_metrics(observed, predicted)
    assert metrics["missed_events"] == 1
    assert metrics["false_events"] == 1
    assert metrics["mae_minutes"] == 7.5

