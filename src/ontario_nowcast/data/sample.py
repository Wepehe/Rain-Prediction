"""Download and crop a bounded MRMS event sample with immutable provenance."""

from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from ..config import load_config
from .manifest import download_immutable, sha256_file
from .mrms import BUCKET, objects_between
from .s3 import S3Object, object_url


def _timestamp_from_key(key: str) -> datetime:
    stamp = key.rsplit("_", 1)[-1].split(".grib2", 1)[0]
    return datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)


def _select_cadence(
    objects: list[S3Object], start: datetime, end: datetime, interval_minutes: int
) -> list[tuple[datetime, S3Object | None]]:
    """Select exact expected cycles and leave gaps explicit instead of phase-shifting the series."""
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be positive")
    by_time = {_timestamp_from_key(obj.key): obj for obj in objects}
    selected = []
    expected = start
    while expected <= end:
        selected.append((expected, by_time.get(expected)))
        expected += timedelta(minutes=interval_minutes)
    return selected


def decode_mrms_crop(path: Path, bbox: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one gzip-compressed MRMS GRIB2 field and crop in its native 0.01° grid."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'data' extra to decode GRIB2") from exc
    west, south, east, north = bbox
    west_360, east_360 = west % 360, east % 360
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with gzip.open(path, "rb") as compressed:
            while chunk := compressed.read(1024 * 1024):
                temporary.write(chunk)
    try:
        with xr.open_dataset(
            temporary_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
        ) as dataset:
            variable = next(iter(dataset.data_vars.values()))
            crop = variable.sel(
                latitude=slice(north, south), longitude=slice(west_360, east_360)
            ).load()
            values = crop.values.astype(np.float32)
            # MRMS reserves -1 for missing and -3 for no radar coverage.
            values[values < 0] = np.nan
            return values, crop.latitude.values, ((crop.longitude.values + 180) % 360) - 180
    finally:
        temporary_path.unlink(missing_ok=True)


def fetch_event(config_path: Path, event_id: str, output_root: Path, interval_minutes: int) -> Path:
    config = load_config(config_path)
    configured_events = (
        list(config.get("sample_events", []))
        + list(config.get("events", []))
        + list(config.get("hard_negative_events", []))
        + list(config.get("development_events", []))
        + list(config.get("final_events", []))
    )
    event = next((item for item in configured_events if item["id"] == event_id), None)
    if event is None:
        raise KeyError(f"unknown sample event: {event_id}")
    start = datetime.fromisoformat(event["start_utc"])
    end = datetime.fromisoformat(event["end_utc"])
    selected = _select_cadence(objects_between(start, end), start, end, interval_minutes)
    raw_dir = output_root / "raw" / "radar" / "mrms" / event_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "metadata" / "downloads.jsonl"
    frames: list[np.ndarray | None] = []
    times, source_files = [], []
    latitudes = longitudes = None
    for position, (expected_time, obj) in enumerate(selected, start=1):
        times.append(expected_time.isoformat())
        if obj is None:
            print(
                f"[{position}/{len(selected)}] missing expected scan {expected_time.isoformat()}",
                flush=True,
            )
            frames.append(None)
            source_files.append("")
            continue
        destination = raw_dir / Path(obj.key).name
        if not destination.exists():
            download_immutable(
                object_url(BUCKET, obj.key),
                destination,
                manifest,
                metadata={
                    "source": "NOAA MRMS",
                    "product": "PrecipRate",
                    "units": "mm h-1",
                    "source_timestamp": _timestamp_from_key(obj.key).isoformat(),
                    "etag": obj.etag,
                },
            )
        elif destination.stat().st_size != obj.size:
            raise RuntimeError(f"existing raw file has wrong size: {destination}")
        print(f"[{position}/{len(selected)}] decoding {destination.name}", flush=True)
        frame, latitudes, longitudes = decode_mrms_crop(destination, event["bbox_wgs84"])
        frames.append(frame)
        source_files.append(obj.key)
    template = next((frame for frame in frames if frame is not None), None)
    if template is None or latitudes is None or longitudes is None:
        raise RuntimeError("event window contains no decodable radar frames")
    complete_frames = [
        frame if frame is not None else np.full_like(template, np.nan) for frame in frames
    ]
    processed = output_root / "processed" / "events" / f"{event_id}.npz"
    processed.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed,
        rate_mm_hr=np.stack(complete_frames),
        times=np.asarray(times),
        latitude=latitudes,
        longitude=longitudes,
        source_files=np.asarray(source_files),
    )
    build_record = {
        "event_id": event_id,
        "created_at": datetime.now(UTC).isoformat(),
        "output": processed.as_posix(),
        "sha256": sha256_file(processed),
        "frames": len(complete_frames),
        "missing_frames": sum(frame is None for frame in frames),
        "bbox_wgs84": event["bbox_wgs84"],
        "cadence_minutes": interval_minutes,
        "source": "NOAA MRMS PrecipRate",
        "source_units": "mm h-1",
        "preprocessing": "native 0.01 degree crop; negative MRMS missing/no-coverage codes to NaN",
    }
    build_manifest = output_root / "metadata" / "dataset_builds.jsonl"
    build_manifest.parent.mkdir(parents=True, exist_ok=True)
    with build_manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(build_record, sort_keys=True) + "\n")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_id")
    parser.add_argument("--config", type=Path, default=Path("configs/data/milestone_1.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--interval-minutes", type=int, default=6)
    args = parser.parse_args()
    print(fetch_event(args.config, args.event_id, args.output_root, args.interval_minutes))


if __name__ == "__main__":
    main()
