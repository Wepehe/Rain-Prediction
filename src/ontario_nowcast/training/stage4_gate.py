"""Stage 4 training-gate validation without model scoring.

This module intentionally performs only data and metadata checks.  It mines
radar objects for the new Stage 4 repair/holdout events, freezes the final
initiation-holdout object manifest, and summarizes whether it is safe to start
the first multimodal ablation.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..data.manifest import sha256_file
from ..events.tracking import build_independent_initiation_catalog
from ..sample_evaluation import _downsample_max

HOLDOUT_SPLIT = "stage4_initiation_holdout"
TILE_SIZE_PIXELS = 128


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _event_entries(config: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in config.get("events", []):
        rows.append({**event, "config_path": config_path.as_posix(), "role": "stage4_event"})
    for event in config.get("hard_negative_events", []):
        rows.append({**event, "config_path": config_path.as_posix(), "role": "hard_negative"})
    return rows


def _axis_downsample(axis: np.ndarray, length: int, factor: int) -> np.ndarray:
    return np.asarray(axis)[: length * factor : factor]


def _tile_bounds(centre: float, size: int, limit: int) -> tuple[int, int]:
    if limit <= size:
        return 0, limit - 1
    start = round(centre) - size // 2
    start = max(0, min(start, limit - size))
    return start, start + size - 1


def _tile_wgs84(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
) -> str:
    lat_values = np.asarray(latitudes)[[y_min, y_max]]
    lon_values = np.asarray(longitudes)[[x_min, x_max]]
    west = float(np.nanmin(lon_values))
    east = float(np.nanmax(lon_values))
    south = float(np.nanmin(lat_values))
    north = float(np.nanmax(lat_values))
    return f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"


def _load_stage4_radar(event_id: str, data_root: Path) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    path = data_root / "processed" / "events" / f"{event_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing processed radar tensor: {path}")
    payload = np.load(path)
    rates_native = payload["rate_mm_hr"]
    factor = 2
    rates = _downsample_max(rates_native, factor)
    latitudes = _axis_downsample(payload["latitude"], rates.shape[1], factor)
    longitudes = _axis_downsample(payload["longitude"], rates.shape[2], factor)
    times = pd.to_datetime(payload["times"], utc=True)
    return rates, times, latitudes, longitudes


def _empty_object_row(event: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "source_event_id": event["id"],
        "object_id": "",
        "split": event.get("split"),
        "event_class": event.get("class"),
        "role": event.get("role"),
        "paired_negative_id": event.get("paired_negative_id", event.get("match_to", "")),
        "representative_issue_time_utc": "",
        "start_time_utc": "",
        "end_time_utc": "",
        "centre_lat": np.nan,
        "centre_lon": np.nan,
        "y_min": np.nan,
        "y_max": np.nan,
        "x_min": np.nan,
        "x_max": np.nan,
        "tile_y_min": np.nan,
        "tile_y_max": np.nan,
        "tile_x_min": np.nan,
        "tile_x_max": np.nan,
        "tile_bbox_wgs84": "",
        "component_count": 0,
        "pixel_count": 0,
        "future_max_mm_hr": np.nan,
        "apparent_initiation": False,
        "advective_entry_like": False,
        "touches_domain_boundary": False,
        "clean_pre_radar_initiation": False,
        "validation_reason": reason,
    }


def mine_event_objects(
    repair_config_path: Path,
    output_dir: Path,
    *,
    data_root: Path = Path("data"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mine Stage 4 object metadata without running any forecast model."""
    config = _load_yaml(repair_config_path)
    rows = []
    summaries = []
    events = _event_entries(config, repair_config_path)
    for event in events:
        rates, times, latitudes, longitudes = _load_stage4_radar(event["id"], data_root)
        catalog = build_independent_initiation_catalog(
            rates,
            times,
            latitudes,
            longitudes,
            interval_minutes=int(config["benchmark"]["cadence_minutes"]),
            resolution_km=float(config["benchmark"]["evaluation_resolution_km"]),
            history_minutes=60,
            horizon_minutes=120,
            rain_threshold=0.1,
            strong_threshold=5.0,
            minimum_pixels=12,
        )
        clean_count = 0
        advective_count = 0
        if catalog.empty:
            rows.append(_empty_object_row(event, "no_independent_initiation_objects_mined"))
        for object_index, item in catalog.reset_index(drop=True).iterrows():
            object_id = str(item["event_id"])
            y_centre = (float(item["y_min"]) + float(item["y_max"])) / 2
            x_centre = (float(item["x_min"]) + float(item["x_max"])) / 2
            tile_y_min, tile_y_max = _tile_bounds(y_centre, TILE_SIZE_PIXELS, rates.shape[1])
            tile_x_min, tile_x_max = _tile_bounds(x_centre, TILE_SIZE_PIXELS, rates.shape[2])
            apparent = bool(item["apparent_initiation"])
            advective = bool(item["advective_entry_like"])
            touches_boundary = bool(item["touches_domain_boundary"])
            clean = (
                event.get("class") == "positive_initiation"
                and apparent
                and not advective
                and not touches_boundary
                and float(item["future_max_mm_hr"]) >= 1.0
            )
            clean_count += int(clean)
            advective_count += int(advective)
            rows.append(
                {
                    "source_event_id": event["id"],
                    "object_id": object_id or f"initiation_track_{object_index + 1:04d}",
                    "split": event.get("split"),
                    "event_class": event.get("class"),
                    "role": event.get("role"),
                    "paired_negative_id": event.get("paired_negative_id", event.get("match_to", "")),
                    "representative_issue_time_utc": item["representative_anchor_time"],
                    "start_time_utc": item["start_time"],
                    "end_time_utc": item["end_time"],
                    "centre_lat": float(item["centre_lat"]),
                    "centre_lon": float(item["centre_lon"]),
                    "y_min": int(item["y_min"]),
                    "y_max": int(item["y_max"]),
                    "x_min": int(item["x_min"]),
                    "x_max": int(item["x_max"]),
                    "tile_y_min": tile_y_min,
                    "tile_y_max": tile_y_max,
                    "tile_x_min": tile_x_min,
                    "tile_x_max": tile_x_max,
                    "tile_bbox_wgs84": _tile_wgs84(
                        latitudes,
                        longitudes,
                        y_min=tile_y_min,
                        y_max=tile_y_max,
                        x_min=tile_x_min,
                        x_max=tile_x_max,
                    ),
                    "component_count": int(item["component_count"]),
                    "pixel_count": int(item["pixel_count"]),
                    "future_max_mm_hr": float(item["future_max_mm_hr"]),
                    "apparent_initiation": apparent,
                    "advective_entry_like": advective,
                    "touches_domain_boundary": touches_boundary,
                    "clean_pre_radar_initiation": clean,
                    "validation_reason": "object_mined",
                }
            )
        summaries.append(
            {
                "event_id": event["id"],
                "split": event.get("split"),
                "event_class": event.get("class"),
                "objects_mined": len(catalog),
                "clean_pre_radar_initiation_objects": int(clean_count),
                "advective_entry_like_objects": int(advective_count),
                "radar_frames": int(rates.shape[0]),
                "evaluation_shape_y": int(rates.shape[1]),
                "evaluation_shape_x": int(rates.shape[2]),
            }
        )
    object_table = pd.DataFrame(rows)
    summary_table = pd.DataFrame(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    object_table.to_csv(output_dir / "event_object_validation.csv", index=False)
    summary_table.to_csv(output_dir / "event_object_summary.csv", index=False)
    return object_table, summary_table


def freeze_final_holdout_manifest(object_table: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Write and hash the Stage 4 final initiation-holdout object manifest."""
    columns = [
        "source_event_id",
        "object_id",
        "representative_issue_time_utc",
        "tile_y_min",
        "tile_y_max",
        "tile_x_min",
        "tile_x_max",
        "tile_bbox_wgs84",
        "event_class",
        "clean_pre_radar_initiation",
        "advective_entry_like",
        "paired_negative_id",
    ]
    manifest = object_table[
        (object_table["split"] == HOLDOUT_SPLIT)
        & (object_table["event_class"] == "positive_initiation")
        & object_table["object_id"].astype(str).ne("")
    ].copy()
    for column in columns:
        if column not in manifest:
            manifest[column] = ""
    manifest = manifest[columns].sort_values(
        ["source_event_id", "representative_issue_time_utc", "object_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "stage4_final_initiation_holdout_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "manifest_path": manifest_path.as_posix(),
        "sha256": sha256_file(manifest_path),
        "rows": len(manifest),
        "clean_pre_radar_initiation_rows": int(manifest["clean_pre_radar_initiation"].sum())
        if len(manifest)
        else 0,
        "created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "scoring_policy": (
            "metadata-only freeze; no multimodal or neural holdout performance was inspected"
        ),
    }
    (output_dir / "stage4_final_initiation_holdout_manifest.sha256.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def summarize_hard_negative_decisions(
    hard_negative_csv: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Copy verifier decisions into a compact gate artifact."""
    if not hard_negative_csv.exists():
        table = pd.DataFrame()
        summary = {
            "source": hard_negative_csv.as_posix(),
            "hard_negative_gate_passed": False,
            "pairs_attempted": 0,
            "pairs_verified": 0,
            "pairs_rejected": 0,
            "reason": "hard_negative_verifier_output_missing",
        }
    else:
        table = pd.read_csv(hard_negative_csv)
        if table.empty:
            pairs_verified = 0
            pairs_rejected = 0
        else:
            pairs_verified = int(table["verified_hard_negative"].astype(bool).sum())
            pairs_rejected = int((~table["verified_hard_negative"].astype(bool)).sum())
        summary = {
            "source": hard_negative_csv.as_posix(),
            "hard_negative_gate_passed": bool(len(table) > 0 and pairs_rejected == 0),
            "pairs_attempted": len(table),
            "pairs_verified": pairs_verified,
            "pairs_rejected": pairs_rejected,
            "rejected_negative_event_ids": sorted(
                table.loc[
                    ~table.get("verified_hard_negative", pd.Series(dtype=bool)).astype(bool),
                    "negative_event_id",
                ]
                .astype(str)
                .tolist()
            )
            if not table.empty and "negative_event_id" in table
            else [],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "hard_negative_decisions.csv", index=False)
    (output_dir / "hard_negative_decisions_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return table, summary


def run_stage4_training_gate(
    repair_config_path: Path,
    output_dir: Path,
    *,
    audit_summary_path: Path = Path("artifacts/stage_4/audit/stage4_audit_summary.json"),
    hard_negative_csv: Path = Path(
        "artifacts/stage_4/evaluation_repair/hard_negative_pair_matching.csv"
    ),
    goes_materialization_summary_path: Path = Path(
        "artifacts/stage_4/materialization/stage4_goes_c13_materialization_summary.json"
    ),
    hrrr_materialization_summary_path: Path = Path(
        "artifacts/stage_4/materialization/stage4_hrrr_materialization_summary.json"
    ),
    radar_goes_tensor_summary_path: Path = Path(
        "artifacts/stage_4/materialization/stage4_radar_goes_tensor_summary.json"
    ),
    radar_goes_visual_summary_path: Path = Path(
        "artifacts/stage_4/visual_sanity/stage4_radar_goes_visual_sanity_summary.json"
    ),
    tiny_overfit_summary_path: Path = Path(
        "artifacts/stage_4/tiny_overfit/stage4_tiny_overfit_summary.json"
    ),
    data_root: Path = Path("data"),
) -> dict[str, Any]:
    """Run metadata-only Stage 4 gate checks and return the gate summary."""
    object_table, object_summary = mine_event_objects(
        repair_config_path, output_dir, data_root=data_root
    )
    holdout_metadata = freeze_final_holdout_manifest(object_table, output_dir)
    _, hard_negative_summary = summarize_hard_negative_decisions(hard_negative_csv, output_dir)
    if audit_summary_path.exists():
        audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
        radar_materialized_gate = audit_summary.get(
            "active_radar_materialized_events",
            audit_summary.get("radar_materialized_events"),
        ) == audit_summary.get("active_events", audit_summary.get("events"))
    else:
        audit_summary = {}
        radar_materialized_gate = False
    if goes_materialization_summary_path.exists():
        goes_summary = json.loads(goes_materialization_summary_path.read_text(encoding="utf-8"))
    else:
        goes_summary = {}
    if hrrr_materialization_summary_path.exists():
        hrrr_summary = json.loads(hrrr_materialization_summary_path.read_text(encoding="utf-8"))
    else:
        hrrr_summary = {}
    if radar_goes_tensor_summary_path.exists():
        radar_goes_tensor_summary = json.loads(
            radar_goes_tensor_summary_path.read_text(encoding="utf-8")
        )
    else:
        radar_goes_tensor_summary = {}
    if radar_goes_visual_summary_path.exists():
        radar_goes_visual_summary = json.loads(
            radar_goes_visual_summary_path.read_text(encoding="utf-8")
        )
    else:
        radar_goes_visual_summary = {}
    if tiny_overfit_summary_path.exists():
        tiny_overfit_summary = json.loads(tiny_overfit_summary_path.read_text(encoding="utf-8"))
    else:
        tiny_overfit_summary = {}
    positive_holdout = object_summary[
        (object_summary["split"] == HOLDOUT_SPLIT)
        & (object_summary["event_class"] == "positive_initiation")
    ]
    clean_holdout_gate = bool(
        not positive_holdout.empty
        and int(positive_holdout["clean_pre_radar_initiation_objects"].sum()) > 0
    )
    radar_goes_gates = {
        "stage31_radar_control_preserved": bool(
            audit_summary.get("frozen_radar_control_files", 0) >= 5
        ),
        "all_radar_windows_materialized": bool(radar_materialized_gate),
        "hard_negatives_verified": bool(hard_negative_summary["hard_negative_gate_passed"]),
        "event_objects_validated": True,
        "final_initiation_holdout_manifest_frozen": bool(holdout_metadata["rows"] > 0),
        "final_holdout_has_clean_pre_radar_objects": clean_holdout_gate,
        "goes_c13_source_files_materialized": bool(
            goes_summary.get("goes_c13_sources_complete", False)
        ),
        "radar_goes_anchor_tensors_materialized": bool(
            radar_goes_tensor_summary.get("complete", False)
            and not radar_goes_tensor_summary.get("truncated_by_max_anchors", True)
        ),
        "radar_goes_visual_sanity_panels_completed": bool(
            radar_goes_visual_summary.get("complete", False)
        ),
        "goes_issue_time_latency_safe": bool(audit_summary.get("availability_records", 0) > 0),
        "tiny_multimodal_overfit_completed": bool(tiny_overfit_summary.get("complete", False)),
    }
    nwp_gates = {
        "actual_multimodal_files_materialized": False,
        "hrrr_nwp_source_files_materialized": bool(
            hrrr_summary.get("hrrr_sources_complete", False)
            and not hrrr_summary.get("truncated_by_max_products", True)
        ),
        "visual_multimodal_sanity_panels_completed": False,
    }
    gates = {**radar_goes_gates, **nwp_gates}
    gate_b_passed = bool(all(radar_goes_gates.values()))
    full_multimodal_gate_passed = bool(all(gates.values()))
    summary = {
        "created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "repair_config": repair_config_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "training_started": False,
        "ablation_b_started": False,
        "gate_passed": gate_b_passed,
        "ablation_b_authorized": gate_b_passed,
        "full_multimodal_gate_passed": full_multimodal_gate_passed,
        "gates": gates,
        "modality_gates": {
            "B_radar_goes": {
                "passed": gate_b_passed,
                "gates": radar_goes_gates,
                "does_not_require": [
                    "full_HRRR_materialization",
                    "NWP_static_tensors",
                    "NWP_inclusive_visual_panels",
                ],
            },
            "C_to_F_nwp_static": {
                "passed": bool(all(nwp_gates.values())),
                "gates": nwp_gates,
            },
        },
        "hard_negative_summary": hard_negative_summary,
        "goes_c13_materialization_summary": goes_summary,
        "hrrr_nwp_materialization_summary": hrrr_summary,
        "radar_goes_tensor_summary": radar_goes_tensor_summary,
        "radar_goes_visual_summary": radar_goes_visual_summary,
        "tiny_overfit_summary": tiny_overfit_summary,
        "holdout_manifest": holdout_metadata,
        "event_object_summary": {
            "events_validated": len(object_summary),
            "objects_mined": int(
                object_summary["objects_mined"].sum() if not object_summary.empty else 0
            ),
            "clean_pre_radar_initiation_objects": int(
                object_summary["clean_pre_radar_initiation_objects"].sum()
                if not object_summary.empty
                else 0
            ),
        },
        "blocked_reason": None
        if gate_b_passed
        else "Stage 4B radar+GOES gate is not open; do not train Ablation B.",
        "nwp_blocked_reason": None
        if full_multimodal_gate_passed
        else "NWP/static gates remain closed; do not train Ablations C-F.",
    }
    (output_dir / "stage4_training_gate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repair-config",
        type=Path,
        default=Path("configs/data/stage_4_evaluation_repair.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4/gate"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=Path("artifacts/stage_4/audit/stage4_audit_summary.json"),
    )
    parser.add_argument(
        "--hard-negative-csv",
        type=Path,
        default=Path("artifacts/stage_4/evaluation_repair/hard_negative_pair_matching.csv"),
    )
    parser.add_argument(
        "--goes-materialization-summary",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_goes_c13_materialization_summary.json"),
    )
    parser.add_argument(
        "--hrrr-materialization-summary",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_hrrr_materialization_summary.json"),
    )
    parser.add_argument(
        "--radar-goes-tensor-summary",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_summary.json"),
    )
    parser.add_argument(
        "--radar-goes-visual-summary",
        type=Path,
        default=Path("artifacts/stage_4/visual_sanity/stage4_radar_goes_visual_sanity_summary.json"),
    )
    parser.add_argument(
        "--tiny-overfit-summary",
        type=Path,
        default=Path("artifacts/stage_4/tiny_overfit/stage4_tiny_overfit_summary.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage4_training_gate(
                args.repair_config,
                args.output_dir,
                audit_summary_path=args.audit_summary,
                hard_negative_csv=args.hard_negative_csv,
                goes_materialization_summary_path=args.goes_materialization_summary,
                hrrr_materialization_summary_path=args.hrrr_materialization_summary,
                radar_goes_tensor_summary_path=args.radar_goes_tensor_summary,
                radar_goes_visual_summary_path=args.radar_goes_visual_summary,
                tiny_overfit_summary_path=args.tiny_overfit_summary,
                data_root=args.data_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
