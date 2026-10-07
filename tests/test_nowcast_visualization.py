import json
from pathlib import Path

import numpy as np

from ontario_nowcast.operational.inference import OCCURRENCE_THRESHOLD
from ontario_nowcast.operational.visualization import (
    LEADS_MINUTES,
    area_summary,
    lead_to_index,
    load_forecast_npz,
    point_series,
    point_summary,
)

PRODUCT = Path("artifacts/operational/smoke_test_output/forecast.npz")


def test_operational_product_loads_with_all_leads():
    product = load_forecast_npz(PRODUCT)
    assert product.pysteps_rate_mm_h.shape[0] == 20
    assert product.residual_probability.shape[0] == 20
    assert LEADS_MINUTES == tuple(range(6, 121, 6))


def test_lead_mapping():
    assert lead_to_index(6) == 0
    assert lead_to_index(30) == 4
    assert lead_to_index(60) == 9
    assert lead_to_index(120) == 19


def test_point_extraction_and_summary_are_direct():
    product = load_forecast_npz(PRODUCT)
    series = point_series(product, 64, 64)
    assert np.array_equal(series["residual_probability"], product.residual_probability[:, 64, 64])
    summary = point_summary(product, 64, 64)
    wet = product.residual_probability[:, 64, 64] >= 0.35
    assert summary["wet_forecast_frames"] == int(wet.sum())
    assert summary["maximum_probability"] == float(product.residual_probability[:, 64, 64].max())


def test_area_summary_uses_unit_interval_not_percent_twice():
    product = load_forecast_npz(PRODUCT)
    summary = area_summary(product, 60)
    probability = product.residual_probability[9]
    assert summary["mean_probability"] == float(probability.mean())
    assert 0 <= summary["mean_probability"] <= 1
    assert summary["wet_area_fraction"] == float((probability >= 0.35).mean())
    assert OCCURRENCE_THRESHOLD == 0.35


def test_demo_is_downstream_only_and_example_is_non_final():
    source = Path("app/nowcast_demo.py").read_text(encoding="utf-8").lower()
    assert "ontario_nowcast.training" not in source
    assert "final_manifest" not in source
    golden = json.loads(Path("tests/fixtures/operational_residual_v1/golden.json").read_text())
    assert golden["split"] == "new_dev"
    assert "final" not in golden["row_id"].lower()
