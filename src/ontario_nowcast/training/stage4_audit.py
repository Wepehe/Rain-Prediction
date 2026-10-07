"""Stage 4 split repair and multimodal availability audit utilities."""

from __future__ import annotations

import argparse
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..data.availability import (
    validate_anchor_table,
    validate_forecast_model_table,
    validate_split_independence,
    validate_train_only_normalization_record,
)
from ..data.manifest import sha256_file


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def _floor_hour(value: pd.Timestamp) -> pd.Timestamp:
    return value.floor("h")


def _file_record(label: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "label": label,
            "path": path.as_posix(),
            "exists": False,
            "sha256": None,
            "bytes": None,
        }
    return {
        "label": label,
        "path": path.as_posix(),
        "exists": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "last_write_time_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz=UTC).isoformat(),
    }


def write_radar_control_freeze_manifest(config: dict[str, Any], output_path: Path) -> list[dict[str, Any]]:
    """Record hashes for the frozen Stage 3.1 radar-only control."""
    frozen = config["frozen_radar_control"]
    files = [
        ("stage31_config", Path(frozen["config"])),
        ("stage31_checkpoint", Path(frozen["checkpoint"])),
        ("stage31_normalization", Path(frozen["normalization"])),
        ("stage31_split_manifest", Path(frozen["split_manifest"])),
        ("stage31_sample_manifest", Path(frozen["sample_manifest"])),
    ]
    for path in frozen.get("metric_outputs", []):
        files.append(("stage31_metric_output", Path(path)))
    records = [_file_record(frozen["label"] + ":" + label, path) for label, path in files]
    missing = [record["path"] for record in records if not record["exists"]]
    if missing:
        raise FileNotFoundError(f"frozen radar control files are missing: {missing}")
    normalization = json.loads(Path(frozen["normalization"]).read_text(encoding="utf-8"))
    train_samples = int(frozen["train_samples"])
    validate_train_only_normalization_record(normalization, expected_train_samples=train_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def _events_from_config(config_path: Path, role: str) -> list[dict[str, Any]]:
    config = _load_yaml(config_path)
    rows = []
    for event in config.get("events", []):
        rows.append({**event, "config_path": config_path.as_posix(), "role": role})
    for event in config.get("hard_negative_events", []):
        rows.append({**event, "config_path": config_path.as_posix(), "role": "hard_negative"})
    return rows


def _active_event_ids(split_manifest: dict[str, Any]) -> set[str]:
    active: set[str] = set()
    for key, value in split_manifest.items():
        if key.endswith("_events") and isinstance(value, list):
            active.update(str(item) for item in value)
    return active


def build_event_inventory(
    config: dict[str, Any],
    output_path: Path,
    *,
    active_event_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Write an auditable inventory of all Stage 4 events and materialization status."""
    rows: list[dict[str, Any]] = []
    data_root = Path(config.get("data_root", "data"))
    for entry in config["event_configs"]:
        rows.extend(_events_from_config(Path(entry["path"]), entry.get("role", "positive")))
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Stage 4 event inventory is empty")
    table["processed_radar_path"] = table["id"].map(
        lambda event_id: (data_root / "processed" / "events" / f"{event_id}.npz").as_posix()
    )
    table["radar_materialized"] = table["processed_radar_path"].map(lambda value: Path(value).exists())
    if active_event_ids is None:
        table["active_in_split_manifest"] = True
    else:
        table["active_in_split_manifest"] = table["id"].astype(str).isin(active_event_ids)
    keep_columns = [
        "id",
        "split",
        "class",
        "role",
        "type_labels",
        "start_utc",
        "end_utc",
        "bbox_wgs84",
        "paired_negative_id",
        "match_to",
        "status",
        "rejection_reason",
        "active_in_split_manifest",
        "radar_materialized",
        "processed_radar_path",
        "rationale",
        "config_path",
    ]
    for column in keep_columns:
        if column not in table:
            table[column] = None
    table = table[keep_columns].sort_values(["split", "role", "id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def _issue_times_for_event(event: pd.Series, cadence_minutes: int) -> list[pd.Timestamp]:
    start = _utc(event["start_utc"]) + pd.Timedelta(minutes=60)
    end = _utc(event["end_utc"]) - pd.Timedelta(minutes=120)
    if end < start:
        start = _utc(event["start_utc"])
        end = _utc(event["end_utc"])
    values = pd.date_range(start, end, freq=f"{cadence_minutes}min", tz="UTC")
    return [pd.Timestamp(value).tz_convert("UTC") for value in values]


def _goes_records(anchor_id: str, issue_time: pd.Timestamp, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    latency = pd.Timedelta(minutes=int(cfg.get("availability_latency_minutes", 10)))
    latest_usable_scan = issue_time - latency
    for channel in cfg["channels"]:
        for lag in cfg["history_lags_minutes"]:
            valid_time = latest_usable_scan - pd.Timedelta(minutes=int(lag))
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "modality": "satellite",
                    "variable": channel,
                    "source": cfg["source_dataset"],
                    "source_timestamp": valid_time.isoformat(),
                    "valid_time_utc": valid_time.isoformat(),
                    "available_time_utc": (valid_time + latency).isoformat(),
                    "spatial_resolution": cfg["spatial_resolution"],
                    "temporal_resolution": cfg["temporal_resolution"],
                    "units": cfg["units"],
                    "missing_fraction": None,
                    "interpolation_method": cfg["interpolation_method"],
                    "normalization_method": cfg["normalization_method"],
                }
            )
    return rows


def _hrrr_cycle(issue_time: pd.Timestamp, availability_lag_minutes: int) -> pd.Timestamp:
    cycle = _floor_hour(issue_time)
    lag = pd.Timedelta(minutes=availability_lag_minutes)
    while cycle + lag > issue_time:
        cycle -= pd.Timedelta(hours=1)
    return cycle


def _nwp_records(anchor_id: str, issue_time: pd.Timestamp, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    cycle = _hrrr_cycle(issue_time, int(cfg.get("availability_lag_minutes", 60)))
    available = cycle + pd.Timedelta(minutes=int(cfg.get("availability_lag_minutes", 60)))
    for variable in cfg["variables"]:
        for offset in cfg["forecast_offsets_minutes"]:
            valid_time = issue_time + pd.Timedelta(minutes=int(offset))
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "modality": "nwp",
                    "variable": variable,
                    "source": cfg["source_dataset"],
                    "source_timestamp": cycle.isoformat(),
                    "valid_time_utc": valid_time.isoformat(),
                    "available_time_utc": available.isoformat(),
                    "model": "HRRR",
                    "model_issue_time_utc": cycle.isoformat(),
                    "model_available_time_utc": available.isoformat(),
                    "model_valid_time_utc": valid_time.isoformat(),
                    "spatial_resolution": cfg["spatial_resolution"],
                    "temporal_resolution": cfg["temporal_resolution"],
                    "units": cfg["units"].get(variable, "unknown"),
                    "missing_fraction": None,
                    "interpolation_method": cfg["interpolation_method"],
                    "normalization_method": cfg["normalization_method"],
                }
            )
    return rows


def _radar_records(anchor_id: str, issue_time: pd.Timestamp, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for lag in cfg["history_lags_minutes"]:
        valid_time = issue_time - pd.Timedelta(minutes=int(lag))
        rows.append(
            {
                "anchor_id": anchor_id,
                "modality": "radar",
                "variable": "precip_rate",
                "source": cfg["source_dataset"],
                "source_timestamp": valid_time.isoformat(),
                "valid_time_utc": valid_time.isoformat(),
                "available_time_utc": valid_time.isoformat(),
                "spatial_resolution": cfg["spatial_resolution"],
                "temporal_resolution": cfg["temporal_resolution"],
                "units": cfg["units"],
                "missing_fraction": None,
                "interpolation_method": cfg["interpolation_method"],
                "normalization_method": cfg["normalization_method"],
            }
        )
    return rows


def _static_records(anchor_id: str, issue_time: pd.Timestamp, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    available = pd.Timestamp(cfg.get("available_time_utc", "2000-01-01T00:00:00Z")).tz_convert("UTC")
    rows = []
    for variable in cfg["variables"]:
        rows.append(
            {
                "anchor_id": anchor_id,
                "modality": "static",
                "variable": variable,
                "source": cfg["source_dataset"],
                "source_timestamp": available.isoformat(),
                "valid_time_utc": issue_time.isoformat(),
                "available_time_utc": available.isoformat(),
                "spatial_resolution": cfg["spatial_resolution"],
                "temporal_resolution": "constant",
                "units": cfg["units"].get(variable, "unitless"),
                "missing_fraction": None,
                "interpolation_method": cfg["interpolation_method"],
                "normalization_method": cfg["normalization_method"],
            }
        )
    return rows


def build_availability_audit(
    config: dict[str, Any], event_inventory: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Build and validate issue-time availability records for Stage 4 anchors."""
    cadence_minutes = int(config["audit"]["anchor_cadence_minutes"])
    max_anchors = int(config["audit"].get("max_anchors_per_event", 4))
    modalities = config["modalities"]
    rows: list[dict[str, Any]] = []
    for _, event in event_inventory.iterrows():
        issue_times = _issue_times_for_event(event, cadence_minutes)[:max_anchors]
        for issue_time in issue_times:
            anchor_id = f"{event['id']}__{issue_time:%Y%m%dT%H%M%SZ}"
            base = {
                "event_id": event["id"],
                "split": event["split"],
                "event_class": event["class"],
                "role": event["role"],
                "forecast_issue_time_utc": issue_time.isoformat(),
            }
            for record in _radar_records(anchor_id, issue_time, modalities["radar"]):
                rows.append({**base, **record})
            for record in _goes_records(anchor_id, issue_time, modalities["satellite"]):
                rows.append({**base, **record})
            for record in _nwp_records(anchor_id, issue_time, modalities["nwp"]):
                rows.append({**base, **record})
            for record in _static_records(anchor_id, issue_time, modalities["static"]):
                rows.append({**base, **record})
    table = pd.DataFrame(rows)
    validate_anchor_table(
        table[
            [
                "anchor_id",
                "modality",
                "source",
                "valid_time_utc",
                "available_time_utc",
                "forecast_issue_time_utc",
            ]
        ]
    )
    nwp = table[table["modality"] == "nwp"].copy()
    validate_forecast_model_table(
        nwp[
            [
                "anchor_id",
                "model",
                "forecast_issue_time_utc",
                "model_issue_time_utc",
                "model_available_time_utc",
                "model_valid_time_utc",
            ]
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def run_stage4_audit(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    split_manifest = _load_yaml(Path(config["split_manifest"]))
    split_map = validate_split_independence(split_manifest)
    active_ids = _active_event_ids(split_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_records = write_radar_control_freeze_manifest(
        config, output_dir / "radar_control_freeze_manifest.json"
    )
    inventory = build_event_inventory(
        config,
        output_dir / "event_inventory.csv",
        active_event_ids=active_ids,
    )
    configured_event_ids = set(inventory["id"].astype(str))
    missing_from_config = sorted(set(split_map) - configured_event_ids)
    if missing_from_config:
        raise ValueError(f"split manifest events missing from Stage 4 event configs: {missing_from_config}")
    active_inventory = inventory[inventory["active_in_split_manifest"]]
    audit = build_availability_audit(
        config,
        active_inventory,
        output_dir / "multimodal_availability_audit.csv",
    )
    summary = {
        "config": config_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "frozen_radar_control_files": len(frozen_records),
        "events": len(inventory),
        "active_events": len(active_inventory),
        "radar_materialized_events": int(inventory["radar_materialized"].sum()),
        "active_radar_materialized_events": int(active_inventory["radar_materialized"].sum()),
        "availability_records": len(audit),
        "splits": inventory.groupby("split").size().to_dict(),
        "active_splits": active_inventory.groupby("split").size().to_dict(),
        "created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "training_started": False,
    }
    (output_dir / "stage4_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/stage_4_multimodal.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4/audit"))
    args = parser.parse_args()
    print(json.dumps(run_stage4_audit(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
