import numpy as np
import pytest

from ontario_nowcast.evaluation.metrics import (
    brier_score,
    categorical_metrics,
    fractions_skill_score,
)


def test_categorical_metrics_known_table() -> None:
    obs = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    metrics = categorical_metrics(obs, pred, 0.5)
    assert metrics["hits"] == metrics["misses"] == metrics["false_alarms"] == 1
    assert metrics["csi"] == pytest.approx(1 / 3)
    assert metrics["f1"] == pytest.approx(0.5)


def test_brier_score_and_probability_validation() -> None:
    assert brier_score(np.array([0, 1]), np.array([0.25, 0.75])) == pytest.approx(0.0625)
    with pytest.raises(ValueError):
        brier_score(np.array([0]), np.array([1.1]))


def test_fss_perfect_empty_forecast() -> None:
    zeros = np.zeros((10, 10))
    assert fractions_skill_score(zeros, zeros, 0.1, 3) == 1.0
