"""Bounded GOES/HRRR/ECCC-station retrieval and timestamp synchronization."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..preprocessing.align import align_nearest
from .manifest import download_immutable
from .s3 import list_public_bucket, object_url

GOES_BUCKET = "noaa-goes16"
GOES_EAST_BUCKETS = ("noaa-goes19", "noaa-goes16")
HRRR_BUCKET = "noaa-hrrr-bdp-pds"
HRRR_FIELDS = (
    ":MSLMA:mean sea level:",
    ":TMP:2 m above ground:",
    ":DPT:2 m above ground:",
    ":UGRD:10 m above ground:",
    ":VGRD:10 m above ground:",
    ":CAPE:surface:",
)
ECCC_TORONTO_STATIONS = {"toronto_pearson": 51459, "toronto_city_centre": 48549}


def _goes_scan_time(key: str) -> datetime:
    match = re.search(r"_s(\d{13})", key)
    if not match:
        raise ValueError(f"cannot parse GOES scan time: {key}")
    return datetime.strptime(match.group(1), "%Y%j%H%M%S").replace(tzinfo=UTC)


def _nearest_goes_channel13(hour: datetime):
    prefix = f"ABI-L2-CMIPC/{hour:%Y}/{hour:%j}/{hour:%H}/"
    objects = [item for item in list_public_bucket(GOES_BUCKET, prefix) if "C13_" in item.key]
    if not objects:
        raise RuntimeError(f"no GOES-16 channel 13 objects under {prefix}")
    return min(objects, key=lambda item: abs(_goes_scan_time(item.key) - hour))


def nearest_goes_east_channel13(hour: datetime):
    """Find the nearest GOES-East ABI channel 13 object, preferring GOES-19 when present."""
    prefix = f"ABI-L2-CMIPC/{hour:%Y}/{hour:%j}/{hour:%H}/"
    errors = []
    for bucket in GOES_EAST_BUCKETS:
        objects = [item for item in list_public_bucket(bucket, prefix) if "C13_" in item.key]
        if objects:
            return bucket, min(objects, key=lambda item: abs(_goes_scan_time(item.key) - hour))
        errors.append(f"{bucket}/{prefix}")
    raise RuntimeError(f"no GOES-East channel 13 objects under: {', '.join(errors)}")


def _hrrr_index(hour: datetime) -> tuple[str, list[tuple[int, int, str]]]:
    base_key = f"hrrr.{hour:%Y%m%d}/conus/hrrr.t{hour:%H}z.wrfsfcf00.grib2"
    index_url = object_url(HRRR_BUCKET, base_key + ".idx")
    response = requests.get(index_url, timeout=60)
    response.raise_for_status()
    lines = [line for line in response.text.splitlines() if line]
    parsed = [(int(line.split(":", 2)[1]), line) for line in lines]
    content_length = int(
        requests.head(object_url(HRRR_BUCKET, base_key), timeout=60).headers["Content-Length"]
    )
    ranges = []
    for position, (start, line) in enumerate(parsed):
        if not any(field in line for field in HRRR_FIELDS):
            continue
        end = parsed[position + 1][0] - 1 if position + 1 < len(parsed) else content_length - 1
        ranges.append((start, end, line))
    if len(ranges) != len(HRRR_FIELDS):
        raise RuntimeError(f"found {len(ranges)} of {len(HRRR_FIELDS)} requested HRRR fields")
    return base_key, ranges


def _download_hrrr_subset(hour: datetime, destination: Path, manifest_path: Path) -> dict:
    base_key, ranges = _hrrr_index(hour)
    url = object_url(HRRR_BUCKET, base_key)
    if destination.exists():
        return {"local_path": destination.as_posix(), "source_timestamp": hour.isoformat()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with destination.open("xb") as handle:
        try:
            for start, end, _ in ranges:
                response = requests.get(
                    url, headers={"Range": f"bytes={start}-{end}"}, timeout=120
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError("HRRR server ignored byte-range request")
                handle.write(response.content)
                digest.update(response.content)
                total += len(response.content)
        except BaseException:
            handle.close()
            destination.unlink(missing_ok=True)
            raise
    record = {
        "source_url": url,
        "source_timestamp": hour.isoformat(),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "local_path": destination.resolve().as_posix(),
        "bytes": total,
        "sha256": digest.hexdigest(),
        "source": "NOAA HRRR analysis",
        "product": "wrfsfcf00 selected GRIB2 messages",
        "byte_ranges": [[start, end] for start, end, _ in ranges],
        "records": [line for _, _, line in ranges],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _download_eccc_station(station_id: int, year: int, month: int, destination: Path, manifest: Path):
    url = (
        "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
        f"format=csv&stationID={station_id}&Year={year}&Month={month}&Day=1&"
        "timeframe=1&submit=Download+Data"
    )
    if not destination.exists():
        download_immutable(
            url,
            destination,
            manifest,
            metadata={
                "source": "ECCC Historical Climate Data",
                "product": "hourly station observations",
                "station_id": station_id,
                "source_time_zone": "local standard time",
            },
        )
    return destination


def fetch_and_synchronize(radar_sample: Path, data_root: Path, artifact_dir: Path) -> Path:
    radar = np.load(radar_sample)
    radar_times = pd.to_datetime(radar["times"], utc=True)
    start, end = radar_times[0].to_pydatetime(), radar_times[-1].to_pydatetime()
    hours = []
    current = start.replace(minute=0, second=0, microsecond=0)
    while current <= end:
        hours.append(current)
        current += timedelta(hours=1)
    manifest = data_root / "metadata" / "downloads.jsonl"

    satellite_records = []
    nwp_records = []
    for hour in hours:
        goes = _nearest_goes_channel13(hour)
        goes_destination = data_root / "raw" / "satellite" / "goes16" / Path(goes.key).name
        if not goes_destination.exists():
            download_immutable(
                object_url(GOES_BUCKET, goes.key),
                goes_destination,
                manifest,
                metadata={
                    "source": "NOAA GOES-16 ABI",
                    "product": "ABI-L2-CMIPC channel 13 brightness temperature",
                    "source_timestamp": _goes_scan_time(goes.key).isoformat(),
                    "etag": goes.etag,
                },
            )
        satellite_records.append((_goes_scan_time(goes.key), goes_destination.as_posix()))
        nwp_destination = data_root / "raw" / "nwp" / "hrrr" / f"hrrr.{hour:%Y%m%d%H}.subset.grib2"
        _download_hrrr_subset(hour, nwp_destination, manifest)
        nwp_records.append((hour, nwp_destination.as_posix()))

    station_paths = [
        _download_eccc_station(
            station_id,
            start.year,
            start.month,
            data_root / "raw" / "stations" / "eccc" / f"{name}_{start:%Y%m}.csv",
            manifest,
        )
        for name, station_id in ECCC_TORONTO_STATIONS.items()
    ]
    station_frames = []
    for path in station_paths:
        frame = pd.read_csv(path)
        # ECCC explicitly labels this field LST: Toronto local standard time is fixed UTC-05:00.
        frame["time_utc"] = pd.to_datetime(frame["Date/Time (LST)"], errors="coerce").dt.tz_localize(
            timezone(timedelta(hours=-5))
        ).dt.tz_convert("UTC")
        station_frames.append(frame)
    station_times = pd.DatetimeIndex(
        sorted(set(pd.concat(station_frames, ignore_index=True)["time_utc"].dropna()))
    )

    satellite_times = pd.DatetimeIndex([record[0] for record in satellite_records])
    nwp_times = pd.DatetimeIndex([record[0] for record in nwp_records])
    sat_index, sat_valid = align_nearest(satellite_times, radar_times, pd.Timedelta("31min"))
    nwp_index, nwp_valid = align_nearest(nwp_times, radar_times, pd.Timedelta("31min"))
    station_index, station_valid = align_nearest(station_times, radar_times, pd.Timedelta("31min"))

    table = pd.DataFrame({"radar_time_utc": radar_times})
    for name, times, indices, valid in (
        ("satellite", satellite_times, sat_index, sat_valid),
        ("nwp", nwp_times, nwp_index, nwp_valid),
        ("station", station_times, station_index, station_valid),
    ):
        matched = pd.Series(pd.NaT, index=table.index, dtype="datetime64[ns, UTC]")
        matched.loc[valid] = times[indices[valid]]
        table[f"{name}_time_utc"] = matched
        table[f"{name}_offset_minutes"] = (
            table[f"{name}_time_utc"] - table["radar_time_utc"]
        ).dt.total_seconds() / 60
        table[f"{name}_available"] = valid
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / "synchronization.csv"
    table.to_csv(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("radar_sample", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/milestone_1/multimodal"))
    args = parser.parse_args()
    print(fetch_and_synchronize(args.radar_sample, args.data_root, args.artifact_dir))


if __name__ == "__main__":
    main()
