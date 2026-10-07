"""Hourly environmental summaries for positive/negative benchmark matching."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..data.manifest import download_immutable
from ..data.multimodal import (
    ECCC_TORONTO_STATIONS,
    _download_eccc_station,
    _download_hrrr_subset,
    _goes_scan_time,
    nearest_goes_east_channel13,
)
from ..data.s3 import object_url
from ..preprocessing.fuse import _interpolate_goes, _station_fields, analysis_grid


def _hours(start: datetime, end: datetime) -> list[datetime]:
    values = []
    current = start.replace(minute=0, second=0, microsecond=0)
    while current <= end:
        values.append(current)
        current += timedelta(hours=1)
    return values


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _event_index(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    events = list(config.get("events", [])) + list(config.get("hard_negative_events", []))
    return {event["id"]: event for event in events}


def _hard_negative_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return explicit and event-table hard-negative entries that have a positive match."""
    entries = [
        entry
        for entry in config.get("hard_negative_events", [])
        if not str(entry.get("status", "")).startswith("rejected")
    ]
    known = {entry["id"] for entry in entries}
    for event in config.get("events", []):
        if (
            event.get("class") == "hard_negative_candidate"
            and "match_to" in event
            and event["id"] not in known
            and not str(event.get("status", "")).startswith("rejected")
        ):
            entries.append(event)
            known.add(event["id"])
    return entries


def _verification_bbox(event: dict[str, Any]) -> list[float]:
    return list(event.get("verification_bbox_wgs84", event["bbox_wgs84"]))


def _safe_nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def _station_paths(event: dict[str, Any], data_root: Path, manifest: Path) -> list[Path]:
    start = datetime.fromisoformat(event["start_utc"])
    return [
        _download_eccc_station(
            station_id,
            start.year,
            start.month,
            data_root / "raw" / "stations" / "eccc" / f"{name}_{start:%Y%m}.csv",
            manifest,
        )
        for name, station_id in ECCC_TORONTO_STATIONS.items()
    ]


def _download_goes(hour: datetime, data_root: Path, manifest: Path) -> Path:
    bucket, goes = nearest_goes_east_channel13(hour)
    destination = data_root / "raw" / "satellite" / bucket / Path(goes.key).name
    if not destination.exists():
        download_immutable(
            object_url(bucket, goes.key),
            destination,
            manifest,
            metadata={
                "source": f"NOAA {bucket}",
                "product": "ABI-L2-CMIPC channel 13 brightness temperature",
                "source_timestamp": _goes_scan_time(goes.key).isoformat(),
                "etag": goes.etag,
            },
        )
    return destination


def _hrrr_native_bbox_fields(path: Path, bbox_wgs84: list[float]) -> dict[str, np.ndarray]:
    import cfgrib

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")
        groups = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    variables = {name: group[name] for group in groups for name in group.data_vars}
    reference = next(iter(variables.values()))
    longitude = ((reference.longitude.values + 180) % 360) - 180
    latitude = reference.latitude.values
    west, south, east, north = bbox_wgs84
    padding = 0.25
    keep = (
        (longitude >= west - padding)
        & (longitude <= east + padding)
        & (latitude >= south - padding)
        & (latitude <= north + padding)
    )
    if not np.any(keep):
        raise RuntimeError(f"no HRRR source points found inside bbox {bbox_wgs84}")
    return {name: variable.values[keep].astype(np.float32) for name, variable in variables.items()}


def summarize_event_environment(
    event: dict[str, Any],
    data_root: Path,
    artifact_dir: Path,
    *,
    grid_resolution_km: float = 20.0,
) -> pd.DataFrame:
    """Fetch and summarize hourly GOES/HRRR/station fields over an event bbox."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / "environment_summary.csv"
    bbox_wgs84 = _verification_bbox(event)
    bbox_key = ",".join(str(value) for value in bbox_wgs84)
    if destination.exists():
        cached = pd.read_csv(destination)
        if "bbox_wgs84" in cached.columns and cached["bbox_wgs84"].astype(str).eq(bbox_key).all():
            return cached
    start = datetime.fromisoformat(event["start_utc"])
    end = datetime.fromisoformat(event["end_utc"])
    manifest = data_root / "metadata" / "downloads.jsonl"
    x, y, longitude, latitude = analysis_grid(tuple(bbox_wgs84), grid_resolution_km * 1000)
    target_x, target_y = np.meshgrid(x, y)
    station_paths = _station_paths(event, data_root, manifest)
    rows = []
    for hour in _hours(start, end):
        goes_path = _download_goes(hour, data_root, manifest)
        hrrr_path = data_root / "raw" / "nwp" / "hrrr" / f"hrrr.{hour:%Y%m%d%H}.subset.grib2"
        _download_hrrr_subset(hour, hrrr_path, manifest)
        c13 = _interpolate_goes(goes_path, longitude, latitude)
        hrrr = _hrrr_native_bbox_fields(hrrr_path, bbox_wgs84)
        stations = _station_fields(
            station_paths, pd.Timestamp(hour).tz_convert("UTC"), target_x, target_y
        )
        wind_speed = np.sqrt(hrrr["u10"] ** 2 + hrrr["v10"] ** 2)
        rows.append(
            {
                "event_id": event["id"],
                "bbox_wgs84": bbox_key,
                "valid_time_utc": hour.isoformat(),
                "cape_mean_j_kg": float(np.nanmean(hrrr["cape"])),
                "cape_p90_j_kg": float(np.nanpercentile(hrrr["cape"], 90)),
                "dewpoint_mean_k": float(np.nanmean(hrrr["d2m"])),
                "c13_mean_k": float(np.nanmean(c13)),
                "c13_p10_k": float(np.nanpercentile(c13, 10)),
                "wind_speed_mean_m_s": float(np.nanmean(wind_speed)),
                "mslp_mean_pa": float(np.nanmean(hrrr["mslma"])),
                "station_dewpoint_mean_c": _safe_nanmean(stations["station_dewpoint_c"]),
                "station_wind_mean_m_s": _safe_nanmean(stations["station_wind_m_s"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(destination, index=False)
    return table


def radar_dry_summary(
    path: Path,
    *,
    bbox_wgs84: list[float] | None = None,
    rain_threshold: float = 0.1,
    heavy_threshold: float = 1.0,
    sparse_mean_wet_fraction: float = 0.01,
    sparse_max_frame_wet_fraction: float = 0.03,
    sparse_mean_heavy_fraction: float = 0.005,
    sparse_max_frame_heavy_fraction: float = 0.02,
) -> dict[str, float | int | bool]:
    payload = np.load(path)
    rates = payload["rate_mm_hr"]
    if bbox_wgs84 is not None:
        latitude = payload["latitude"]
        longitude = payload["longitude"]
        west, south, east, north = bbox_wgs84
        y_keep = np.where((latitude >= south) & (latitude <= north))[0]
        x_keep = np.where((longitude >= west) & (longitude <= east))[0]
        if len(y_keep) == 0 or len(x_keep) == 0:
            raise RuntimeError(f"verification bbox outside radar crop: {bbox_wgs84}")
        rates = rates[:, y_keep.min() : y_keep.max() + 1, x_keep.min() : x_keep.max() + 1]
    finite = np.isfinite(rates)
    wet = finite & (rates >= rain_threshold)
    heavy = finite & (rates >= heavy_threshold)
    frame_wet = []
    frame_heavy = []
    for frame, frame_finite in zip(wet, finite, strict=True):
        if np.any(frame_finite):
            frame_wet.append(float(np.mean(frame[frame_finite])))
        else:
            frame_wet.append(float("nan"))
    for frame, frame_finite in zip(heavy, finite, strict=True):
        if np.any(frame_finite):
            frame_heavy.append(float(np.mean(frame[frame_finite])))
        else:
            frame_heavy.append(float("nan"))
    finite_rates = rates[finite]
    mean_wet = float(np.mean(wet[finite])) if np.any(finite) else float("nan")
    max_frame_wet = float(np.nanmax(frame_wet)) if frame_wet else float("nan")
    mean_heavy = float(np.mean(heavy[finite])) if np.any(finite) else float("nan")
    max_frame_heavy = float(np.nanmax(frame_heavy)) if frame_heavy else float("nan")
    strict_dry = bool(np.any(finite) and np.nanmax(rates) < rain_threshold)
    sparse_dry = bool(
        np.any(finite)
        and mean_wet <= sparse_mean_wet_fraction
        and max_frame_wet <= sparse_max_frame_wet_fraction
        and mean_heavy <= sparse_mean_heavy_fraction
        and max_frame_heavy <= sparse_max_frame_heavy_fraction
    )
    return {
        "frames": int(rates.shape[0]),
        "verification_bbox_wgs84": ",".join(str(value) for value in bbox_wgs84)
        if bbox_wgs84 is not None
        else "",
        "finite_fraction": float(np.mean(finite)),
        "max_rate_mm_hr": float(np.nanmax(rates)) if np.any(finite) else float("nan"),
        "p99_rate_mm_hr": float(np.nanpercentile(finite_rates, 99))
        if np.any(finite)
        else float("nan"),
        "p999_rate_mm_hr": float(np.nanpercentile(finite_rates, 99.9))
        if np.any(finite)
        else float("nan"),
        "wet_pixel_fraction": mean_wet,
        "max_frame_wet_pixel_fraction": max_frame_wet,
        "heavy_pixel_fraction": mean_heavy,
        "max_frame_heavy_pixel_fraction": max_frame_heavy,
        "strict_dry_window": strict_dry,
        "sparse_dry_window": sparse_dry,
        "dry_window": sparse_dry,
    }


def aggregate_environment(table: pd.DataFrame) -> dict[str, float]:
    fields = [
        "cape_mean_j_kg",
        "cape_p90_j_kg",
        "dewpoint_mean_k",
        "c13_mean_k",
        "c13_p10_k",
        "wind_speed_mean_m_s",
        "mslp_mean_pa",
    ]
    return {field: float(table[field].mean()) for field in fields}


def pair_similarity(
    positive_summary: pd.DataFrame,
    negative_summary: pd.DataFrame,
    positive_event: dict[str, Any],
    negative_event: dict[str, Any],
) -> dict[str, Any]:
    pos = aggregate_environment(positive_summary)
    neg = aggregate_environment(negative_summary)
    row: dict[str, Any] = {
        "positive_event_id": positive_event["id"],
        "negative_event_id": negative_event["id"],
        "same_season_label": bool(
            set(positive_event.get("type_labels", []))
            & set(negative_event.get("type_labels", []))
            & {"winter", "spring", "summer", "autumn", "cool_season"}
        ),
    }
    pos_start = pd.Timestamp(positive_event["start_utc"])
    neg_start = pd.Timestamp(negative_event["start_utc"])
    hour_delta = abs(pos_start.hour + pos_start.minute / 60 - (neg_start.hour + neg_start.minute / 60))
    row["time_of_day_difference_hours"] = float(min(hour_delta, 24 - hour_delta))
    standardized = []
    for field, pos_value in pos.items():
        neg_value = neg[field]
        series = pd.concat([positive_summary[field], negative_summary[field]], ignore_index=True)
        scale = float(series.std(ddof=0))
        diff = float(neg_value - pos_value)
        std_diff = diff / scale if scale > 1e-6 else 0.0
        row[f"positive_{field}"] = pos_value
        row[f"negative_{field}"] = neg_value
        row[f"std_diff_{field}"] = std_diff
        standardized.append(abs(std_diff))
    row["mean_abs_standardized_difference"] = float(np.mean(standardized))
    return row


def verify_hard_negative_pairs(
    config_path: Path,
    data_root: Path,
    artifact_root: Path,
    *,
    fetch_radar,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    from ..data.sample import fetch_event

    config = _load_config(config_path)
    events = _event_index(config)
    rows = []
    failures = []
    negatives = _hard_negative_entries(config)
    if max_pairs is not None:
        negatives = negatives[:max_pairs]
    for index, negative in enumerate(negatives, start=1):
        positive = events[negative["match_to"]]
        pair_dir = artifact_root / "hn_pairs" / negative["id"]
        try:
            print(f"[{index}/{len(negatives)}] verifying {negative['id']}", flush=True)
            neg_radar = data_root / "processed" / "events" / f"{negative['id']}.npz"
            if not neg_radar.exists() and fetch_radar:
                fetch_event(config_path, negative["id"], data_root, config["benchmark"]["cadence_minutes"])
            if not neg_radar.exists():
                raise FileNotFoundError(f"missing negative radar tensor: {neg_radar}")
            dry = radar_dry_summary(neg_radar, bbox_wgs84=_verification_bbox(negative))
            if not dry["dry_window"]:
                print(
                    f"[{index}/{len(negatives)}] rejected by radar: {negative['id']}",
                    flush=True,
                )
                rows.append(
                    {
                        "positive_event_id": positive["id"],
                        "negative_event_id": negative["id"],
                        "verified_hard_negative": False,
                        "rejection_reason": "radar_precipitation_too_extensive",
                        **dry,
                    }
                )
                continue
            print(
                f"[{index}/{len(negatives)}] radar pass; matching GOES/HRRR: {negative['id']}",
                flush=True,
            )
            pos_env = summarize_event_environment(
                positive, data_root, pair_dir / "positive"
            )
            neg_env = summarize_event_environment(
                negative, data_root, pair_dir / "negative"
            )
            rows.append(
                {
                    **pair_similarity(pos_env, neg_env, positive, negative),
                    "verified_hard_negative": True,
                    "rejection_reason": "",
                    **dry,
                }
            )
        except Exception as exc:  # noqa: BLE001 - verifier should continue across pairs
            failures.append(
                {
                    "positive_event_id": positive["id"],
                    "negative_event_id": negative["id"],
                    "error": str(exc),
                }
            )
    artifact_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(artifact_root / "hard_negative_pair_matching.csv", index=False)
    pd.DataFrame(failures).to_csv(artifact_root / "hard_negative_failures.csv", index=False)
    return {
        "pairs_attempted": len(negatives),
        "pairs_verified": int(sum(bool(row.get("verified_hard_negative")) for row in rows)),
        "pairs_failed": len(failures),
        "dry_pairs": int(sum(bool(row["dry_window"]) for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/milestone_1_5.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/milestone_1_5"))
    parser.add_argument("--fetch-radar", action="store_true")
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_hard_negative_pairs(
                args.config,
                args.data_root,
                args.artifact_root,
                fetch_radar=args.fetch_radar,
                max_pairs=args.max_pairs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
