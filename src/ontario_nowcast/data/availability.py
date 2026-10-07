"""Issue-time availability checks for benchmark anchors."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InputRecord:
    modality: str
    valid_time_utc: pd.Timestamp
    available_time_utc: pd.Timestamp
    source: str


@dataclass(frozen=True)
class AnchorRecord:
    anchor_id: str
    forecast_issue_time_utc: pd.Timestamp
    inputs: tuple[InputRecord, ...]


def _utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware UTC-compatible: {value}")
    return timestamp.tz_convert("UTC")


def validate_issue_time_safe(anchor: AnchorRecord) -> None:
    """Reject inputs that would not have been available when the forecast was issued."""
    issue_time = _utc_timestamp(anchor.forecast_issue_time_utc)
    for record in anchor.inputs:
        available_time = _utc_timestamp(record.available_time_utc)
        if available_time > issue_time:
            raise ValueError(
                f"future-information leakage for {anchor.anchor_id}: "
                f"{record.modality} from {record.source} available at "
                f"{available_time.isoformat()} after issue {issue_time.isoformat()}"
            )


def validate_anchor_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate a flat anchor manifest and return a copy with normalized UTC timestamps.

    Required columns are anchor_id, modality, source, valid_time_utc,
    available_time_utc, and forecast_issue_time_utc.
    """
    required = {
        "anchor_id",
        "modality",
        "source",
        "valid_time_utc",
        "available_time_utc",
        "forecast_issue_time_utc",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"anchor table missing required columns: {sorted(missing)}")
    normalized = table.copy()
    for column in ("valid_time_utc", "available_time_utc", "forecast_issue_time_utc"):
        normalized[column] = pd.to_datetime(normalized[column], utc=True)
    leaks = normalized["available_time_utc"] > normalized["forecast_issue_time_utc"]
    if leaks.any():
        offenders = normalized.loc[leaks, ["anchor_id", "modality", "available_time_utc"]]
        raise ValueError(f"future-information leakage detected: {offenders.to_dict('records')}")
    return normalized


def validate_forecast_model_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate issue-time-safe NWP inputs while allowing future valid times.

    A numerical weather prediction field is safe when the model cycle was issued
    and available before the nowcast issue time. Its valid time may be at, before,
    or after the nowcast issue time because an operational forecaster would have
    access to forecasts from an already-issued cycle.
    """
    required = {
        "anchor_id",
        "model",
        "forecast_issue_time_utc",
        "model_issue_time_utc",
        "model_available_time_utc",
        "model_valid_time_utc",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"forecast model table missing required columns: {sorted(missing)}")
    normalized = table.copy()
    for column in (
        "forecast_issue_time_utc",
        "model_issue_time_utc",
        "model_available_time_utc",
        "model_valid_time_utc",
    ):
        normalized[column] = pd.to_datetime(normalized[column], utc=True)
    future_cycle = normalized["model_issue_time_utc"] > normalized["forecast_issue_time_utc"]
    unavailable_cycle = normalized["model_available_time_utc"] > normalized["forecast_issue_time_utc"]
    if future_cycle.any() or unavailable_cycle.any():
        offenders = normalized.loc[
            future_cycle | unavailable_cycle,
            [
                "anchor_id",
                "model",
                "forecast_issue_time_utc",
                "model_issue_time_utc",
                "model_available_time_utc",
                "model_valid_time_utc",
            ],
        ]
        raise ValueError(f"future NWP cycle leakage detected: {offenders.to_dict('records')}")
    return normalized


def validate_split_independence(split_manifest: dict[str, list[str] | object]) -> dict[str, str]:
    """Return event->split mapping after checking whole-event disjointness."""
    result: dict[str, str] = {}
    for key, value in split_manifest.items():
        if not key.endswith("_events") or not isinstance(value, list):
            continue
        split = key.removesuffix("_events")
        for event_id in value:
            if not isinstance(event_id, str):
                raise TypeError(f"event identifiers must be strings in {key}")
            if event_id in result:
                raise ValueError(
                    f"event appears in more than one split: {event_id} "
                    f"({result[event_id]}, {split})"
                )
            result[event_id] = split
    return result


def validate_train_only_normalization_record(
    record: dict[str, object], *, expected_train_samples: int | None = None
) -> None:
    """Reject normalization metadata that is not explicitly train-only."""
    if record.get("source_split") != "train":
        raise ValueError("normalization statistics must be marked as source_split='train'")
    if expected_train_samples is not None and int(record.get("train_samples", -1)) != int(
        expected_train_samples
    ):
        raise ValueError(
            "normalization train sample count does not match the frozen training manifest"
        )


def validate_holdout_not_scored(
    evaluation_log: pd.DataFrame,
    *,
    holdout_split: str = "stage4_initiation_holdout",
    procedure_frozen: bool = False,
) -> None:
    """Prevent accidental Stage 4 holdout scoring before the procedure is frozen."""
    if procedure_frozen or evaluation_log.empty:
        return
    required = {"split", "model"}
    missing = required - set(evaluation_log.columns)
    if missing:
        raise ValueError(f"evaluation log missing required columns: {sorted(missing)}")
    touched = evaluation_log["split"].astype(str).eq(holdout_split)
    if touched.any():
        offenders = evaluation_log.loc[touched, ["split", "model"]]
        raise ValueError(
            f"{holdout_split} was scored before Stage 4 procedure freeze: "
            f"{offenders.to_dict('records')}"
        )
