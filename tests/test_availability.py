import pandas as pd
import pytest

from ontario_nowcast.data.availability import (
    AnchorRecord,
    InputRecord,
    validate_anchor_table,
    validate_issue_time_safe,
)


def test_issue_time_validator_accepts_available_inputs() -> None:
    issue = pd.Timestamp("2024-07-16T15:00:00Z")
    anchor = AnchorRecord(
        anchor_id="anchor_001",
        forecast_issue_time_utc=issue,
        inputs=(
            InputRecord("radar", issue, issue, "MRMS"),
            InputRecord(
                "satellite",
                pd.Timestamp("2024-07-16T14:50:00Z"),
                pd.Timestamp("2024-07-16T14:58:00Z"),
                "GOES ABI",
            ),
        ),
    )
    validate_issue_time_safe(anchor)


def test_issue_time_validator_rejects_future_available_inputs() -> None:
    anchor = AnchorRecord(
        anchor_id="anchor_002",
        forecast_issue_time_utc=pd.Timestamp("2024-07-16T15:00:00Z"),
        inputs=(
            InputRecord(
                "nwp",
                pd.Timestamp("2024-07-16T15:00:00Z"),
                pd.Timestamp("2024-07-16T15:15:00Z"),
                "HRRR",
            ),
        ),
    )
    with pytest.raises(ValueError, match="future-information leakage"):
        validate_issue_time_safe(anchor)


def test_anchor_table_rejects_future_leakage() -> None:
    table = pd.DataFrame(
        {
            "anchor_id": ["a"],
            "modality": ["radar"],
            "source": ["MRMS"],
            "valid_time_utc": ["2024-01-01T00:00:00Z"],
            "available_time_utc": ["2024-01-01T00:06:00Z"],
            "forecast_issue_time_utc": ["2024-01-01T00:00:00Z"],
        }
    )
    with pytest.raises(ValueError, match="future-information leakage"):
        validate_anchor_table(table)
