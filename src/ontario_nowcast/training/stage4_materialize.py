"""Materialize Stage 4 multimodal source files from the issue-time audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

from ..data.manifest import download_immutable, sha256_file
from ..data.multimodal import _goes_scan_time, nearest_goes_east_channel13
from ..data.s3 import object_url
from ..preprocessing.fuse import _interpolate_goes
from ..sample_evaluation import _downsample_max

COOLING_PATTERN = re.compile(r"cooling_tendency_(\d+)min")
TILE_SIZE_PIXELS = 128
HRRR_BUCKET = "noaa-hrrr-bdp-pds"
HRRR_VARIABLE_MESSAGES = {
    "TMP_2m": ("wrfsfc", ":TMP:2 m above ground:"),
    "DPT_2m": ("wrfsfc", ":DPT:2 m above ground:"),
    "UGRD_10m": ("wrfsfc", ":UGRD:10 m above ground:"),
    "VGRD_10m": ("wrfsfc", ":VGRD:10 m above ground:"),
    "MSLP": ("wrfsfc", ":MSLMA:mean sea level:"),
    "CAPE_surface": ("wrfsfc", ":CAPE:surface:"),
    "CIN_surface": ("wrfsfc", ":CIN:surface:"),
    "PWAT_column": ("wrfsfc", ":PWAT:entire atmosphere (considered as a single layer):"),
    "TMP_850hPa": ("wrfprs", ":TMP:850 mb:"),
    "RH_850hPa": ("wrfprs", ":RH:850 mb:"),
    "UGRD_850hPa": ("wrfprs", ":UGRD:850 mb:"),
    "VGRD_850hPa": ("wrfprs", ":VGRD:850 mb:"),
    "TMP_700hPa": ("wrfprs", ":TMP:700 mb:"),
    "RH_700hPa": ("wrfprs", ":RH:700 mb:"),
    "UGRD_700hPa": ("wrfprs", ":UGRD:700 mb:"),
    "VGRD_700hPa": ("wrfprs", ":VGRD:700 mb:"),
    "TMP_500hPa": ("wrfprs", ":TMP:500 mb:"),
    "UGRD_500hPa": ("wrfprs", ":UGRD:500 mb:"),
    "VGRD_500hPa": ("wrfprs", ":VGRD:500 mb:"),
    "VVEL_700hPa": ("wrfprs", ":VVEL:700 mb:"),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _active_event_ids(split_manifest: dict[str, Any]) -> set[str]:
    active: set[str] = set()
    for key, value in split_manifest.items():
        if key.endswith("_events") and isinstance(value, list):
            active.update(str(item) for item in value)
    return active


def _normalize_utc(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat()


def _tile_bounds(centre: float, size: int, limit: int) -> tuple[int, int]:
    if limit <= size:
        return 0, limit - 1
    start = round(centre) - size // 2
    start = max(0, min(start, limit - size))
    return start, start + size - 1


def _cooling_tendency(current_c13_k: np.ndarray, baseline_c13_k: np.ndarray) -> np.ndarray:
    """Return positive values where cloud tops cooled between baseline and current scans."""
    return (baseline_c13_k - current_c13_k).astype(np.float32)


def _goes_required_scan_records(audit: pd.DataFrame) -> pd.DataFrame:
    """Return unique requested C13 source times, including cooling baselines."""
    satellite = audit[audit["modality"].astype(str).eq("satellite")].copy()
    rows: list[dict[str, str]] = []
    for _, record in satellite.iterrows():
        valid_time = pd.Timestamp(record["valid_time_utc"]).tz_convert("UTC")
        variable = str(record["variable"])
        base = {
            "event_id": str(record["event_id"]),
            "split": str(record["split"]),
            "anchor_id": str(record["anchor_id"]),
            "forecast_issue_time_utc": pd.Timestamp(
                record["forecast_issue_time_utc"]
            ).isoformat(),
            "variable": variable,
        }
        rows.append(
            {
                **base,
                "required_source_time_utc": valid_time.isoformat(),
                "source_role": "c13_current_or_history",
            }
        )
        match = COOLING_PATTERN.search(variable)
        if match:
            baseline = valid_time - pd.Timedelta(minutes=int(match.group(1)))
            rows.append(
                {
                    **base,
                    "required_source_time_utc": baseline.isoformat(),
                    "source_role": f"c13_cooling_baseline_{match.group(1)}min",
                }
            )
    table = pd.DataFrame(rows)
    return (
        table.drop_duplicates(
            ["event_id", "split", "anchor_id", "required_source_time_utc", "source_role"]
        )
        .sort_values(["event_id", "anchor_id", "required_source_time_utc", "source_role"])
        .reset_index(drop=True)
    )


def _hrrr_forecast_hour(model_issue_time: pd.Timestamp, valid_time: pd.Timestamp) -> int:
    """Return the integer HRRR forecast hour linking a model cycle to a valid time."""
    delta_hours = (
        valid_time.tz_convert("UTC") - model_issue_time.tz_convert("UTC")
    ).total_seconds() / 3600
    rounded = round(delta_hours)
    if abs(delta_hours - rounded) > 1e-6 or rounded < 0:
        raise ValueError(
            "HRRR source materialization expects non-negative whole-hour forecast offsets; "
            f"got {delta_hours:g} hours"
        )
    return int(rounded)


def _hrrr_product_key(model_issue_time: pd.Timestamp, forecast_hour: int, product: str) -> str:
    cycle = model_issue_time.tz_convert("UTC")
    return (
        f"hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z."
        f"{product}f{forecast_hour:02d}.grib2"
    )


def _hrrr_required_product_records(audit: pd.DataFrame) -> pd.DataFrame:
    """Return unique Stage 4 HRRR products and message selectors required by the audit."""
    nwp = audit[audit["modality"].astype(str).eq("nwp")].copy()
    rows: list[dict[str, Any]] = []
    for _, record in nwp.iterrows():
        variable = str(record["variable"])
        if variable not in HRRR_VARIABLE_MESSAGES:
            raise KeyError(f"no HRRR message selector configured for Stage 4 variable: {variable}")
        product, message_selector = HRRR_VARIABLE_MESSAGES[variable]
        model_issue_time = pd.Timestamp(record["model_issue_time_utc"]).tz_convert("UTC")
        valid_time = pd.Timestamp(record["model_valid_time_utc"]).tz_convert("UTC")
        forecast_hour = _hrrr_forecast_hour(model_issue_time, valid_time)
        product_key = _hrrr_product_key(model_issue_time, forecast_hour, product)
        rows.append(
            {
                "event_id": str(record["event_id"]),
                "split": str(record["split"]),
                "anchor_id": str(record["anchor_id"]),
                "forecast_issue_time_utc": pd.Timestamp(
                    record["forecast_issue_time_utc"]
                ).isoformat(),
                "variable": variable,
                "product": product,
                "message_selector": message_selector,
                "model_issue_time_utc": model_issue_time.isoformat(),
                "model_valid_time_utc": valid_time.isoformat(),
                "forecast_hour": forecast_hour,
                "bucket": HRRR_BUCKET,
                "product_key": product_key,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id",
                "split",
                "anchor_id",
                "forecast_issue_time_utc",
                "variable",
                "product",
                "message_selector",
                "model_issue_time_utc",
                "model_valid_time_utc",
                "forecast_hour",
                "bucket",
                "product_key",
            ]
        )
    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            [
                "event_id",
                "split",
                "anchor_id",
                "variable",
                "model_issue_time_utc",
                "model_valid_time_utc",
            ]
        )
        .sort_values(["model_issue_time_utc", "forecast_hour", "product", "variable"])
        .reset_index(drop=True)
    )


def materialize_goes_c13_sources(
    audit_csv: Path,
    output_dir: Path,
    *,
    data_root: Path = Path("data"),
    config_path: Path = Path("configs/experiments/stage_4_multimodal.yaml"),
    active_only: bool = True,
    max_source_times: int | None = None,
) -> dict[str, Any]:
    """Download/prove local GOES-East ABI C13 files needed by Stage 4 anchors."""
    audit = pd.read_csv(audit_csv)
    if active_only:
        config = _load_yaml(config_path)
        active_ids = _active_event_ids(_load_yaml(Path(config["split_manifest"])))
        audit = audit[audit["event_id"].astype(str).isin(active_ids)].copy()
    required = _goes_required_scan_records(audit)
    all_unique_times = (
        pd.to_datetime(required["required_source_time_utc"], utc=True)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    unique_times = list(all_unique_times)
    if max_source_times is not None:
        unique_times = unique_times[:max_source_times]
    manifest = data_root / "metadata" / "downloads.jsonl"
    rows = []
    for index, requested in enumerate(unique_times, start=1):
        requested_ts = pd.Timestamp(requested).to_pydatetime()
        bucket, obj = nearest_goes_east_channel13(requested_ts)
        scan_time = _goes_scan_time(obj.key)
        destination = data_root / "raw" / "satellite" / bucket / Path(obj.key).name
        if not destination.exists():
            print(
                f"[{index}/{len(unique_times)}] downloading {bucket}/{Path(obj.key).name}",
                flush=True,
            )
            download_immutable(
                object_url(bucket, obj.key),
                destination,
                manifest,
                metadata={
                    "source": f"NOAA {bucket}",
                    "product": "ABI-L2-CMIPC channel 13 brightness temperature",
                    "source_timestamp": scan_time.isoformat(),
                    "etag": obj.etag,
                    "stage": "stage_4",
                },
            )
        else:
            print(
                f"[{index}/{len(unique_times)}] cached {bucket}/{Path(obj.key).name}",
                flush=True,
            )
        rows.append(
            {
                "requested_source_time_utc": pd.Timestamp(requested).isoformat(),
                "actual_source_time_utc": scan_time.isoformat(),
                "bucket": bucket,
                "key": obj.key,
                "local_path": destination.as_posix(),
                "exists": destination.exists(),
                "bytes": destination.stat().st_size if destination.exists() else None,
                "sha256": sha256_file(destination) if destination.exists() else None,
                "time_offset_minutes": (
                    pd.Timestamp(scan_time) - pd.Timestamp(requested)
                ).total_seconds()
                / 60,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = pd.DataFrame(rows)
    required.to_csv(output_dir / "stage4_goes_c13_required_sources.csv", index=False)
    sources.to_csv(output_dir / "stage4_goes_c13_materialized_sources.csv", index=False)
    materialized_times = set(sources["requested_source_time_utc"].astype(str)) if len(sources) else set()
    full_requested_times = {pd.Timestamp(time).isoformat() for time in all_unique_times}
    complete = (
        max_source_times is None
        and full_requested_times <= materialized_times
        and len(sources) == len(all_unique_times)
    )
    summary = {
        "created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "audit_csv": audit_csv.as_posix(),
        "active_only": active_only,
        "required_anchor_source_records": len(required),
        "unique_required_source_times": len(all_unique_times),
        "attempted_unique_source_times": len(unique_times),
        "materialized_unique_source_times": int(sources["exists"].sum()) if len(sources) else 0,
        "truncated_by_max_source_times": max_source_times is not None,
        "goes_c13_sources_complete": bool(complete and len(sources) == len(unique_times)),
        "output_dir": output_dir.as_posix(),
        "training_started": False,
    }
    (output_dir / "stage4_goes_c13_materialization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _hrrr_index(product_key: str) -> tuple[int, list[tuple[int, str]]]:
    index_url = object_url(HRRR_BUCKET, product_key + ".idx")
    response = requests.get(index_url, timeout=60)
    response.raise_for_status()
    parsed = [
        (int(line.split(":", 2)[1]), line)
        for line in response.text.splitlines()
        if line
    ]
    content_length = int(
        requests.head(object_url(HRRR_BUCKET, product_key), timeout=60).headers["Content-Length"]
    )
    return content_length, parsed


def _hrrr_message_ranges(
    product_key: str,
    selectors: dict[str, str],
) -> list[dict[str, Any]]:
    content_length, parsed = _hrrr_index(product_key)
    rows = []
    for variable, selector in selectors.items():
        matches = [
            (position, start, line)
            for position, (start, line) in enumerate(parsed)
            if selector in line
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one HRRR message for {variable} selector {selector!r} "
                f"in {product_key}, found {len(matches)}"
            )
        position, start, line = matches[0]
        end = parsed[position + 1][0] - 1 if position + 1 < len(parsed) else content_length - 1
        rows.append(
            {
                "variable": variable,
                "message_selector": selector,
                "byte_start": start,
                "byte_end": end,
                "idx_line": line,
            }
        )
    return rows


def _download_hrrr_subset(
    product_key: str,
    message_ranges: list[dict[str, Any]],
    destination: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    url = object_url(HRRR_BUCKET, product_key)
    if destination.exists():
        return {
            "source_url": url,
            "local_path": destination.resolve().as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "cached": True,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with destination.open("xb") as handle:
        try:
            for message in sorted(message_ranges, key=lambda row: int(row["byte_start"])):
                response = requests.get(
                    url,
                    headers={"Range": f"bytes={message['byte_start']}-{message['byte_end']}"},
                    timeout=120,
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"HRRR server ignored byte-range request for {product_key}")
                handle.write(response.content)
                digest.update(response.content)
                total += len(response.content)
        except BaseException:
            handle.close()
            destination.unlink(missing_ok=True)
            raise
    record = {
        "source_url": url,
        "retrieved_at": pd.Timestamp.now(tz=UTC).isoformat(),
        "local_path": destination.resolve().as_posix(),
        "bytes": total,
        "sha256": digest.hexdigest(),
        "source": "NOAA HRRR",
        "product": "selected Stage 4 GRIB2 messages",
        "bucket": HRRR_BUCKET,
        "product_key": product_key,
        "byte_ranges": [
            [int(message["byte_start"]), int(message["byte_end"])] for message in message_ranges
        ],
        "records": [str(message["idx_line"]) for message in message_ranges],
        "cached": False,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _hrrr_spotcheck_rows(path: Path, *, max_messages: int = 6) -> list[dict[str, Any]]:
    import eccodes  # type: ignore[import-untyped]

    rows = []
    with path.open("rb") as handle:
        while len(rows) < max_messages:
            message = eccodes.codes_grib_new_from_file(handle)
            if message is None:
                break
            try:
                values = eccodes.codes_get_values(message)
                rows.append(
                    {
                        "local_path": path.as_posix(),
                        "short_name": eccodes.codes_get(message, "shortName"),
                        "type_of_level": eccodes.codes_get(message, "typeOfLevel"),
                        "level": eccodes.codes_get(message, "level"),
                        "grid_type": eccodes.codes_get(message, "gridType"),
                        "nx": eccodes.codes_get(message, "Nx"),
                        "ny": eccodes.codes_get(message, "Ny"),
                        "finite_fraction": float(np.mean(np.isfinite(values))),
                        "min": float(np.nanmin(values)),
                        "max": float(np.nanmax(values)),
                    }
                )
            finally:
                eccodes.codes_release(message)
    return rows


def materialize_hrrr_nwp_sources(
    audit_csv: Path,
    output_dir: Path,
    *,
    data_root: Path = Path("data"),
    config_path: Path = Path("configs/experiments/stage_4_multimodal.yaml"),
    active_only: bool = True,
    max_products: int | None = None,
    variables: list[str] | None = None,
) -> dict[str, Any]:
    """Download/prove local HRRR subset files needed by Stage 4 NWP audit records."""
    audit = pd.read_csv(audit_csv)
    if active_only:
        config = _load_yaml(config_path)
        active_ids = _active_event_ids(_load_yaml(Path(config["split_manifest"])))
        audit = audit[audit["event_id"].astype(str).isin(active_ids)].copy()
    required = _hrrr_required_product_records(audit)
    if variables:
        unknown = sorted(set(variables) - set(HRRR_VARIABLE_MESSAGES))
        if unknown:
            raise ValueError(f"unknown requested HRRR variables: {unknown}")
        required = required[required["variable"].astype(str).isin(variables)].copy()
    products = (
        required[["bucket", "product_key", "product", "model_issue_time_utc", "forecast_hour"]]
        .drop_duplicates()
        .sort_values(["model_issue_time_utc", "forecast_hour", "product"])
        .reset_index(drop=True)
    )
    all_product_keys = products["product_key"].astype(str).tolist()
    if max_products is not None:
        products = products.iloc[:max_products].copy()
    manifest_path = data_root / "metadata" / "downloads.jsonl"
    rows = []
    failures = []
    range_rows = []
    spotcheck_rows = []

    def materialize_product(index: int, product: pd.Series) -> dict[str, Any]:
        product_key = str(product["product_key"])
        try:
            subset = required[required["product_key"].astype(str).eq(product_key)]
            selectors = dict(
                sorted(
                    {
                        str(row["variable"]): str(row["message_selector"])
                        for row in subset.to_dict("records")
                    }.items()
                )
            )
            print(
                f"[{index + 1}/{len(products)}] materializing HRRR {product_key} "
                f"({len(selectors)} variables)",
                flush=True,
            )
            message_ranges = _hrrr_message_ranges(product_key, selectors)
            destination = (
                data_root
                / "raw"
                / "nwp"
                / "hrrr"
                / "stage4"
                / product_key.replace("/conus/", "/")
            ).with_suffix(".stage4_subset.grib2")
            record = _download_hrrr_subset(
                product_key,
                message_ranges,
                destination,
                manifest_path,
            )
            return {
                "row": {
                    "bucket": HRRR_BUCKET,
                    "product_key": product_key,
                    "local_path": record["local_path"],
                    "exists": Path(record["local_path"]).exists(),
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                    "cached": record["cached"],
                    "variables": json.dumps(sorted(selectors)),
                    "message_count": len(message_ranges),
                },
                "ranges": [{"product_key": product_key, **message} for message in message_ranges],
            }
        except Exception as exc:  # noqa: BLE001 - record materialization failures
            return {"failure": {"product_key": product_key, "error": str(exc)}}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(materialize_product, int(index), product): str(product["product_key"])
            for index, product in products.iterrows()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if "failure" in result:
                failures.append(result["failure"])
            else:
                rows.append(result["row"])
                range_rows.extend(result["ranges"])
            if completed == 1 or completed % 10 == 0:
                print(f"[hrrr-materialize] completed {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: row["product_key"])
    range_rows.sort(key=lambda row: (row["product_key"], row["variable"]))
    for row in rows[:4]:
        spotcheck_rows.extend(_hrrr_spotcheck_rows(Path(row["local_path"]), max_messages=3))
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = pd.DataFrame(rows)
    failure_table = pd.DataFrame(failures)
    required.to_csv(output_dir / "stage4_hrrr_required_sources.csv", index=False)
    pd.DataFrame(range_rows).to_csv(output_dir / "stage4_hrrr_message_ranges.csv", index=False)
    materialized.to_csv(output_dir / "stage4_hrrr_materialized_sources.csv", index=False)
    failure_table.to_csv(output_dir / "stage4_hrrr_source_failures.csv", index=False)
    pd.DataFrame(spotcheck_rows).to_csv(output_dir / "stage4_hrrr_decode_spotcheck.csv", index=False)
    materialized_keys = set(materialized["product_key"].astype(str)) if len(materialized) else set()
    complete = (
        max_products is None
        and set(all_product_keys) <= materialized_keys
        and len(failure_table) == 0
    )
    summary = {
        "created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "audit_csv": audit_csv.as_posix(),
        "active_only": active_only,
        "required_nwp_records": len(required),
        "variables": sorted(required["variable"].astype(str).unique()),
        "unique_required_hrrr_products": len(all_product_keys),
        "attempted_hrrr_products": len(products),
        "materialized_hrrr_products": int(materialized["exists"].sum()) if len(materialized) else 0,
        "failures": len(failure_table),
        "truncated_by_max_products": max_products is not None,
        "hrrr_sources_complete": bool(complete),
        "output_dir": output_dir.as_posix(),
        "training_started": False,
    }
    (output_dir / "stage4_hrrr_materialization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _event_config_index(stage4_config_path: Path) -> dict[str, dict[str, Any]]:
    config = _load_yaml(stage4_config_path)
    rows = list(config.get("events", [])) + list(config.get("hard_negative_events", []))
    return {str(event["id"]): event for event in rows}


def _source_manifest_map(materialized_csv: Path) -> dict[str, dict[str, Any]]:
    table = pd.read_csv(materialized_csv)
    return {
        _normalize_utc(record["requested_source_time_utc"]): record
        for record in table.to_dict("records")
    }


def _load_event_radar(event_id: str, data_root: Path) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    payload = np.load(data_root / "processed" / "events" / f"{event_id}.npz")
    rates = _downsample_max(payload["rate_mm_hr"], 2)
    latitudes = payload["latitude"][: rates.shape[1] * 2 : 2]
    longitudes = payload["longitude"][: rates.shape[2] * 2 : 2]
    times = pd.to_datetime(payload["times"], utc=True)
    return rates, times, latitudes, longitudes


def _nearest_axis_index(axis: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(axis, dtype=float) - float(value))))


def _tile_for_anchor(
    event_id: str,
    event: dict[str, Any],
    *,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    ny: int,
    nx: int,
) -> tuple[int, int, int, int, str]:
    bbox = event.get("verification_bbox_wgs84", event.get("bbox_wgs84"))
    if bbox:
        west, south, east, north = [float(value) for value in bbox]
        centre_lon = (west + east) / 2
        centre_lat = (south + north) / 2
        centre_y = _nearest_axis_index(latitudes, centre_lat)
        centre_x = _nearest_axis_index(longitudes, centre_lon)
        tile_reason = "verification_or_event_bbox_centre"
    else:
        del event_id
        centre_y = ny // 2
        centre_x = nx // 2
        tile_reason = "radar_crop_centre"
    y0, y1 = _tile_bounds(centre_y, TILE_SIZE_PIXELS, ny)
    x0, x1 = _tile_bounds(centre_x, TILE_SIZE_PIXELS, nx)
    return y0, y1, x0, x1, tile_reason


def _frame_at_time(rates: np.ndarray, times: pd.DatetimeIndex, valid_time: pd.Timestamp) -> np.ndarray:
    matches = np.where(times == valid_time)[0]
    if len(matches) == 0:
        raise KeyError(f"radar time not present in processed tensor: {valid_time.isoformat()}")
    return rates[int(matches[0])]


def _goes_tile(
    source_record: dict[str, Any],
    tile_longitude: np.ndarray,
    tile_latitude: np.ndarray,
) -> np.ndarray:
    path = Path(str(source_record["local_path"]))
    if not path.exists():
        raise FileNotFoundError(f"missing materialized GOES source: {path}")
    return _interpolate_goes(path, tile_longitude, tile_latitude)


def materialize_radar_goes_anchor_tensors(
    audit_csv: Path,
    output_dir: Path,
    *,
    data_root: Path = Path("data"),
    config_path: Path = Path("configs/experiments/stage_4_multimodal.yaml"),
    stage4_event_config: Path = Path("configs/data/stage_4_evaluation_repair.yaml"),
    goes_sources_csv: Path = Path(
        "artifacts/stage_4/materialization/stage4_goes_c13_materialized_sources.csv"
    ),
    max_anchors: int | None = None,
) -> dict[str, Any]:
    """Build aligned radar + GOES C13 tensors for active Stage 4 audit anchors.

    These tensors are intended for data-gate checks, visual sanity panels, and
    tiny-overfit validation.  They do not start model training.
    """
    config = _load_yaml(config_path)
    active_ids = _active_event_ids(_load_yaml(Path(config["split_manifest"])))
    event_index = _event_config_index(stage4_event_config)
    source_map = _source_manifest_map(goes_sources_csv)
    audit = pd.read_csv(audit_csv)
    audit = audit[audit["event_id"].astype(str).isin(active_ids)].copy()
    anchors = (
        audit[["anchor_id", "event_id", "split", "event_class", "forecast_issue_time_utc"]]
        .drop_duplicates()
        .sort_values(["split", "event_id", "forecast_issue_time_utc"])
        .reset_index(drop=True)
    )
    if max_anchors is not None:
        anchors = anchors.iloc[:max_anchors].copy()

    radar_lags = [int(value) for value in config["modalities"]["radar"]["history_lags_minutes"]]
    satellite_lags = [
        int(value) for value in config["modalities"]["satellite"]["history_lags_minutes"]
    ]
    cooling_minutes = [10, 20, 30, 60]
    latency = pd.Timedelta(
        minutes=int(config["modalities"]["satellite"].get("availability_latency_minutes", 10))
    )
    rows = []
    failures = []
    output_root = output_dir / "radar_goes_tensors"
    for index, anchor in anchors.iterrows():
        anchor_id = str(anchor["anchor_id"])
        event_id = str(anchor["event_id"])
        try:
            print(f"[{index + 1}/{len(anchors)}] materializing tensor {anchor_id}", flush=True)
            rates, times, latitudes, longitudes = _load_event_radar(event_id, data_root)
            issue_time = pd.Timestamp(anchor["forecast_issue_time_utc"]).tz_convert("UTC")
            event = event_index.get(event_id, {"id": event_id})
            y0, y1, x0, x1, tile_reason = _tile_for_anchor(
                event_id,
                event,
                latitudes=latitudes,
                longitudes=longitudes,
                ny=rates.shape[1],
                nx=rates.shape[2],
            )
            tile_lat_axis = latitudes[y0 : y1 + 1]
            tile_lon_axis = longitudes[x0 : x1 + 1]
            tile_lon, tile_lat = np.meshgrid(tile_lon_axis, tile_lat_axis)
            radar_history = []
            radar_valid_times = []
            for lag in radar_lags:
                valid_time = issue_time - pd.Timedelta(minutes=lag)
                frame = _frame_at_time(rates, times, valid_time)
                radar_history.append(frame[y0 : y1 + 1, x0 : x1 + 1])
                radar_valid_times.append(valid_time.isoformat())
            target_frames = []
            target_times = []
            for lead in range(6, 121, 6):
                valid_time = issue_time + pd.Timedelta(minutes=lead)
                frame = _frame_at_time(rates, times, valid_time)
                target_frames.append(frame[y0 : y1 + 1, x0 : x1 + 1])
                target_times.append(valid_time.isoformat())
            latest_usable_scan = issue_time - latency
            raw_c13 = []
            raw_times = []
            actual_raw_times = []
            cooling_by_window = {window: [] for window in cooling_minutes}
            cooling_baseline_times = {window: [] for window in cooling_minutes}
            for lag in satellite_lags:
                valid_time = latest_usable_scan - pd.Timedelta(minutes=lag)
                source = source_map[_normalize_utc(valid_time)]
                current = _goes_tile(source, tile_lon, tile_lat)
                raw_c13.append(current)
                raw_times.append(valid_time.isoformat())
                actual_raw_times.append(_normalize_utc(source["actual_source_time_utc"]))
                for window in cooling_minutes:
                    baseline_time = valid_time - pd.Timedelta(minutes=window)
                    baseline_source = source_map[_normalize_utc(baseline_time)]
                    baseline = _goes_tile(baseline_source, tile_lon, tile_lat)
                    cooling_by_window[window].append(_cooling_tendency(current, baseline))
                    cooling_baseline_times[window].append(baseline_time.isoformat())
            satellite_channels = [np.stack(raw_c13).astype(np.float32)]
            satellite_channel_names = ["ABI_C13_brightness_temperature"]
            for window in cooling_minutes:
                satellite_channels.append(np.stack(cooling_by_window[window]).astype(np.float32))
                satellite_channel_names.append(f"ABI_C13_cooling_tendency_{window}min")
            satellite = np.stack(satellite_channels, axis=0)
            radar = np.stack(radar_history).astype(np.float32)
            target = np.stack(target_frames).astype(np.float32)
            tensor_path = output_root / str(anchor["split"]) / f"{anchor_id}.npz"
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                tensor_path,
                radar_rate_mm_hr=radar,
                radar_valid_mask=np.isfinite(radar).astype(np.float32),
                target_rate_mm_hr=target,
                target_valid_mask=np.isfinite(target).astype(np.float32),
                goes=np.nan_to_num(satellite, nan=0.0).astype(np.float32),
                goes_valid_mask=np.isfinite(satellite).astype(np.float32),
                latitude=tile_lat.astype(np.float32),
                longitude=tile_lon.astype(np.float32),
                radar_valid_times_utc=np.asarray(radar_valid_times),
                target_valid_times_utc=np.asarray(target_times),
                goes_requested_times_utc=np.asarray(raw_times),
                goes_actual_source_times_utc=np.asarray(actual_raw_times),
                goes_cooling_baseline_times_json=np.asarray(json.dumps(cooling_baseline_times)),
                satellite_channel_names=np.asarray(satellite_channel_names),
                forecast_issue_time_utc=np.asarray(issue_time.isoformat()),
                tile_indices=np.asarray([y0, y1, x0, x1], dtype=np.int32),
                tile_reason=np.asarray(tile_reason),
                cooling_sign_convention=np.asarray(
                    "baseline_c13_k_minus_current_c13_k; positive means colder cloud tops"
                ),
            )
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "event_id": event_id,
                    "split": anchor["split"],
                    "event_class": anchor["event_class"],
                    "forecast_issue_time_utc": issue_time.isoformat(),
                    "tensor_path": tensor_path.as_posix(),
                    "sha256": sha256_file(tensor_path),
                    "radar_shape": "x".join(str(value) for value in radar.shape),
                    "goes_shape": "x".join(str(value) for value in satellite.shape),
                    "target_shape": "x".join(str(value) for value in target.shape),
                    "radar_missing_fraction": float(np.mean(~np.isfinite(radar))),
                    "goes_missing_fraction": float(np.mean(~np.isfinite(satellite))),
                    "target_missing_fraction": float(np.mean(~np.isfinite(target))),
                    "tile_y_min": y0,
                    "tile_y_max": y1,
                    "tile_x_min": x0,
                    "tile_x_max": x1,
                    "tile_reason": tile_reason,
                }
            )
        except Exception as exc:  # noqa: BLE001 - record all tensor failures
            failures.append(
                {
                    "anchor_id": anchor_id,
                    "event_id": event_id,
                    "split": anchor["split"],
                    "error": str(exc),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    failure_table = pd.DataFrame(failures)
    manifest.to_csv(output_dir / "stage4_radar_goes_tensor_manifest.csv", index=False)
    failure_table.to_csv(output_dir / "stage4_radar_goes_tensor_failures.csv", index=False)
    summary = {
        "created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "audit_csv": audit_csv.as_posix(),
        "anchors_requested": len(anchors),
        "anchors_materialized": len(manifest),
        "failures": len(failure_table),
        "complete": bool(len(anchors) > 0 and len(manifest) == len(anchors) and not failures),
        "truncated_by_max_anchors": max_anchors is not None,
        "output_dir": output_dir.as_posix(),
        "training_started": False,
    }
    (output_dir / "stage4_radar_goes_tensor_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("goes-sources", "hrrr-nwp-sources", "radar-goes-tensors"),
        default="goes-sources",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("artifacts/stage_4/audit/multimodal_availability_audit.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4/materialization"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/stage_4_multimodal.yaml"),
    )
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--max-source-times", type=int)
    parser.add_argument("--max-products", type=int)
    parser.add_argument("--variable", action="append", dest="variables")
    parser.add_argument("--stage4-event-config", type=Path, default=Path("configs/data/stage_4_evaluation_repair.yaml"))
    parser.add_argument(
        "--goes-sources-csv",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_goes_c13_materialized_sources.csv"),
    )
    parser.add_argument("--max-anchors", type=int)
    args = parser.parse_args()
    if args.mode == "goes-sources":
        summary = materialize_goes_c13_sources(
            args.audit_csv,
            args.output_dir,
            data_root=args.data_root,
            config_path=args.config,
            active_only=not args.include_inactive,
            max_source_times=args.max_source_times,
        )
    elif args.mode == "hrrr-nwp-sources":
        summary = materialize_hrrr_nwp_sources(
            args.audit_csv,
            args.output_dir,
            data_root=args.data_root,
            config_path=args.config,
            active_only=not args.include_inactive,
            max_products=args.max_products,
            variables=args.variables,
        )
    else:
        summary = materialize_radar_goes_anchor_tensors(
            args.audit_csv,
            args.output_dir,
            data_root=args.data_root,
            config_path=args.config,
            stage4_event_config=args.stage4_event_config,
            goes_sources_csv=args.goes_sources_csv,
            max_anchors=args.max_anchors,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
