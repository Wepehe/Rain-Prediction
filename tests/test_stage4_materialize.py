import numpy as np
import pandas as pd

from ontario_nowcast.training.stage4_materialize import (
    _cooling_tendency,
    _goes_required_scan_records,
    _hrrr_forecast_hour,
    _hrrr_product_key,
    _hrrr_required_product_records,
    _normalize_utc,
)


def test_goes_required_sources_include_cooling_baseline() -> None:
    audit = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "split": ["train", "train"],
            "anchor_id": ["a1", "a1"],
            "forecast_issue_time_utc": [
                "2026-07-27T18:00:00Z",
                "2026-07-27T18:00:00Z",
            ],
            "modality": ["satellite", "satellite"],
            "variable": [
                "ABI_C13_brightness_temperature",
                "ABI_C13_cooling_tendency_30min",
            ],
            "valid_time_utc": ["2026-07-27T17:50:00Z", "2026-07-27T17:50:00Z"],
        }
    )

    result = _goes_required_scan_records(audit)

    assert "2026-07-27T17:50:00+00:00" in set(result["required_source_time_utc"])
    assert "2026-07-27T17:20:00+00:00" in set(result["required_source_time_utc"])
    assert "c13_cooling_baseline_30min" in set(result["source_role"])


def test_c13_cooling_positive_means_cloud_top_became_colder() -> None:
    current = np.array([[240.0, 260.0]], dtype=np.float32)
    baseline = np.array([[250.0, 255.0]], dtype=np.float32)

    result = _cooling_tendency(current, baseline)

    assert result[0, 0] == 10.0
    assert result[0, 1] == -5.0


def test_normalize_utc_produces_stable_iso_keys() -> None:
    assert _normalize_utc("2026-07-27T13:50:00-04:00") == "2026-07-27T17:50:00+00:00"


def test_hrrr_forecast_hour_and_product_key_are_stable() -> None:
    model_issue = pd.Timestamp("2026-07-27T12:00:00Z")
    valid = pd.Timestamp("2026-07-27T15:00:00Z")

    assert _hrrr_forecast_hour(model_issue, valid) == 3
    assert (
        _hrrr_product_key(model_issue, 3, "wrfprs")
        == "hrrr.20260727/conus/hrrr.t12z.wrfprsf03.grib2"
    )


def test_hrrr_required_product_records_map_stage4_variables() -> None:
    audit = pd.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "split": ["train", "train"],
            "anchor_id": ["a1", "a1"],
            "forecast_issue_time_utc": [
                "2026-07-27T13:00:00Z",
                "2026-07-27T13:00:00Z",
            ],
            "modality": ["nwp", "nwp"],
            "variable": ["TMP_2m", "VVEL_700hPa"],
            "model_issue_time_utc": [
                "2026-07-27T12:00:00Z",
                "2026-07-27T12:00:00Z",
            ],
            "model_valid_time_utc": [
                "2026-07-27T13:00:00Z",
                "2026-07-27T15:00:00Z",
            ],
        }
    )

    result = _hrrr_required_product_records(audit)

    assert {
        ("TMP_2m", "wrfsfc", 1, "hrrr.20260727/conus/hrrr.t12z.wrfsfcf01.grib2"),
        ("VVEL_700hPa", "wrfprs", 3, "hrrr.20260727/conus/hrrr.t12z.wrfprsf03.grib2"),
    } == set(
        zip(
            result["variable"],
            result["product"],
            result["forecast_hour"],
            result["product_key"],
            strict=True,
        )
    )
