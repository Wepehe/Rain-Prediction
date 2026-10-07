import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ontario_nowcast.data.availability import (
    validate_forecast_model_table,
    validate_holdout_not_scored,
    validate_split_independence,
    validate_train_only_normalization_record,
)
from ontario_nowcast.training.stage4_audit import run_stage4_audit


def test_stage4_split_manifest_keeps_repair_and_holdout_disjoint() -> None:
    with open("configs/experiments/stage_4_split_manifest.yaml", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    split_map = validate_split_independence(manifest)
    assert split_map["stage4_init_holdout_sep02_2026_southern_ontario"] == (
        "stage4_initiation_holdout"
    )
    assert split_map["stage4_dev_initiation_2026_06_18"] == "stage4_dev_repair"
    assert not set(manifest["stage4_dev_repair_events"]) & set(
        manifest["stage4_initiation_holdout_events"]
    )


def test_forecast_model_valid_time_may_be_future_if_cycle_is_available() -> None:
    table = pd.DataFrame(
        {
            "anchor_id": ["a"],
            "model": ["HRRR"],
            "forecast_issue_time_utc": ["2026-07-27T18:00:00Z"],
            "model_issue_time_utc": ["2026-07-27T17:00:00Z"],
            "model_available_time_utc": ["2026-07-27T17:45:00Z"],
            "model_valid_time_utc": ["2026-07-27T20:00:00Z"],
        }
    )
    result = validate_forecast_model_table(table)
    assert result.loc[0, "model_valid_time_utc"] > result.loc[0, "forecast_issue_time_utc"]


def test_forecast_model_rejects_unavailable_future_cycle() -> None:
    table = pd.DataFrame(
        {
            "anchor_id": ["a"],
            "model": ["HRRR"],
            "forecast_issue_time_utc": ["2026-07-27T18:00:00Z"],
            "model_issue_time_utc": ["2026-07-27T18:00:00Z"],
            "model_available_time_utc": ["2026-07-27T18:45:00Z"],
            "model_valid_time_utc": ["2026-07-27T19:00:00Z"],
        }
    )
    with pytest.raises(ValueError, match="future NWP cycle leakage"):
        validate_forecast_model_table(table)


def test_normalization_must_be_train_only() -> None:
    validate_train_only_normalization_record(
        {"source_split": "train", "train_samples": 3, "mean": 0.1, "std": 0.2},
        expected_train_samples=3,
    )
    with pytest.raises(ValueError, match="source_split='train'"):
        validate_train_only_normalization_record({"source_split": "dev"})


def test_stage4_initiation_holdout_cannot_be_scored_before_freeze() -> None:
    evaluation_log = pd.DataFrame(
        {
            "split": ["stage4_initiation_holdout"],
            "model": ["radar_plus_satellite"],
        }
    )
    with pytest.raises(ValueError, match="before Stage 4 procedure freeze"):
        validate_holdout_not_scored(evaluation_log, procedure_frozen=False)
    validate_holdout_not_scored(evaluation_log, procedure_frozen=True)


def test_stage4_audit_writes_no_training_artifacts(tmp_path) -> None:
    summary = run_stage4_audit(
        Path("configs/experiments/stage_4_multimodal.yaml"),
        tmp_path,
    )
    assert summary["training_started"] is False
    assert summary["frozen_radar_control_files"] >= 5
    assert (tmp_path / "radar_control_freeze_manifest.json").exists()
    assert (tmp_path / "event_inventory.csv").exists()
    assert (tmp_path / "multimodal_availability_audit.csv").exists()
    manifest = json.loads((tmp_path / "radar_control_freeze_manifest.json").read_text())
    assert all(record["exists"] for record in manifest)
