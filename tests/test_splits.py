import pandas as pd
import pytest

from ontario_nowcast.events.splits import chronological_event_split


def test_whole_events_receive_one_chronological_split() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "start_time": ["2021-01-01T00:00Z", "2022-01-01T00:00Z", "2023-01-01T00:00Z"],
            "end_time": ["2021-01-02T00:00Z", "2022-01-02T00:00Z", "2023-01-02T00:00Z"],
        }
    )
    result = chronological_event_split(
        events,
        train_before=pd.Timestamp("2021-12-31T00:00Z"),
        validation_before=pd.Timestamp("2022-12-31T00:00Z"),
    )
    assert result["split"].tolist() == ["train", "validation", "test"]


def test_event_crossing_boundary_is_rejected() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["storm"],
            "start_time": ["2021-12-30T00:00Z"],
            "end_time": ["2022-01-02T00:00Z"],
        }
    )
    with pytest.raises(ValueError, match="crosses"):
        chronological_event_split(
            events,
            train_before=pd.Timestamp("2022-01-01T00:00Z"),
            validation_before=pd.Timestamp("2023-01-01T00:00Z"),
        )
