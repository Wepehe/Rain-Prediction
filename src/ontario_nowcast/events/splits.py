"""Leakage-resistant chronological splitting at whole-event granularity."""

from __future__ import annotations

import pandas as pd


def chronological_event_split(
    events: pd.DataFrame,
    *,
    train_before: pd.Timestamp,
    validation_before: pd.Timestamp,
) -> pd.DataFrame:
    required = {"event_id", "start_time", "end_time"}
    if not required.issubset(events.columns):
        raise ValueError(f"events must contain {sorted(required)}")
    if events["event_id"].duplicated().any():
        raise ValueError("each event_id must occur exactly once before splitting")
    result = events.copy()
    result["start_time"] = pd.to_datetime(result["start_time"], utc=True)
    result["end_time"] = pd.to_datetime(result["end_time"], utc=True)
    if (result["end_time"] < result["start_time"]).any():
        raise ValueError("an event ends before it starts")
    train_before = pd.Timestamp(train_before)
    validation_before = pd.Timestamp(validation_before)
    if train_before.tzinfo is None or validation_before.tzinfo is None:
        raise ValueError("split boundaries must be timezone-aware")
    if validation_before <= train_before:
        raise ValueError("validation boundary must follow training boundary")
    crosses_boundary = (
        (result["start_time"] < train_before) & (result["end_time"] >= train_before)
    ) | (
        (result["start_time"] < validation_before)
        & (result["end_time"] >= validation_before)
    )
    if crosses_boundary.any():
        raise ValueError("a weather event crosses a split boundary")
    result["split"] = "test"
    result.loc[result["end_time"] < validation_before, "split"] = "validation"
    result.loc[result["end_time"] < train_before, "split"] = "train"
    return result

