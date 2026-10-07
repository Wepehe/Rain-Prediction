"""Live NOAA MRMS ingestion for the frozen Residual V1 operational interface."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import requests

from ..data.mrms import BUCKET, daily_objects
from ..data.s3 import S3Object, object_url
from ..data.sample import _timestamp_from_key, decode_mrms_crop
from ..preprocessing.fuse import _interpolate_regular_latlon, analysis_grid
from ..preprocessing.grid import transform_coordinates

DOMAIN_BBOX_WGS84 = (-84.8, 41.5, -75.5, 46.5)
RESOLUTION_METRES = 2_000.0
TILE_PIXELS = 128
TILE_SPAN_METRES = (TILE_PIXELS - 1) * RESOLUTION_METRES
HISTORY_FRAMES = 10
CADENCE_MINUTES = 6
MIN_AVAILABLE_FRAMES = 9
DEFAULT_CACHE = Path(os.environ.get("NOWCAST_CACHE_DIR", "artifacts/operational/live_cache"))


class LiveDataUnavailable(RuntimeError):
    """Raised when an honest live radar history cannot be assembled."""


@dataclass(frozen=True)
class TileDefinition:
    x: np.ndarray
    y: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    point_x_index: int
    point_y_index: int
    requested_longitude: float
    requested_latitude: float

    @property
    def identity(self) -> str:
        value = f"{self.x[0]:.0f}:{self.y[0]:.0f}:{RESOLUTION_METRES:.0f}:{TILE_PIXELS}"
        return hashlib.sha256(value.encode()).hexdigest()[:20]


@dataclass(frozen=True)
class LiveRadarHistory:
    history_rate: np.ndarray
    timestamps: tuple[datetime, ...]
    validity_mask: np.ndarray
    tile: TileDefinition
    issue_time: datetime
    source_keys: tuple[str, ...]
    missing_timestamps: tuple[datetime, ...]


def supported_domain() -> dict[str, object]:
    x, y, _, _ = analysis_grid(DOMAIN_BBOX_WGS84, RESOLUTION_METRES)
    return {"bbox_wgs84": DOMAIN_BBOX_WGS84, "x_min": float(x.min()), "x_max": float(x.max()),
            "y_min": float(y.min()), "y_max": float(y.max()), "crs": "EPSG:3978"}


def location_in_domain(latitude: float, longitude: float) -> bool:
    west, south, east, north = DOMAIN_BBOX_WGS84
    return west <= longitude <= east and south <= latitude <= north


def tile_for_location(latitude: float, longitude: float) -> TileDefinition:
    if not location_in_domain(latitude, longitude):
        raise ValueError("This location is outside the current Residual V1 forecast domain.")
    domain_x, domain_y, _, _ = analysis_grid(DOMAIN_BBOX_WGS84, RESOLUTION_METRES)
    point_x, point_y = transform_coordinates(longitude, latitude, source_crs="EPSG:4326",
                                              destination_crs="EPSG:3978")
    x0 = round((float(point_x) - TILE_SPAN_METRES / 2) / RESOLUTION_METRES) * RESOLUTION_METRES
    y0 = round((float(point_y) - TILE_SPAN_METRES / 2) / RESOLUTION_METRES) * RESOLUTION_METRES
    x0 = min(max(x0, float(domain_x.min())), float(domain_x.max()) - TILE_SPAN_METRES)
    y0 = min(max(y0, float(domain_y.min())), float(domain_y.max()) - TILE_SPAN_METRES)
    x = x0 + np.arange(TILE_PIXELS) * RESOLUTION_METRES
    y = y0 + np.arange(TILE_PIXELS) * RESOLUTION_METRES
    grid_x, grid_y = np.meshgrid(x, y)
    tile_lon, tile_lat = transform_coordinates(grid_x, grid_y, source_crs="EPSG:3978",
                                               destination_crs="EPSG:4326")
    ix = int(np.argmin(np.abs(x - float(point_x))))
    iy = int(np.argmin(np.abs(y - float(point_y))))
    return TileDefinition(x, y, tile_lon, tile_lat, ix, iy, longitude, latitude)


def latest_radar_objects(now: datetime | None = None,
                         listing: Callable[[datetime], list[S3Object]] = daily_objects) -> list[S3Object]:
    now = (now or datetime.now(UTC)).astimezone(UTC)
    objects: list[S3Object] = []
    for offset in (1, 0):
        objects.extend(listing(now - timedelta(days=offset)))
    available = [obj for obj in objects if _timestamp_from_key(obj.key) <= now]
    if not available:
        raise LiveDataUnavailable("No recent MRMS precipitation-rate scans are available.")
    return sorted(available, key=lambda obj: _timestamp_from_key(obj.key))


def regular_history(objects: list[S3Object]) -> list[tuple[datetime, S3Object | None]]:
    if not objects:
        raise LiveDataUnavailable("No MRMS objects were supplied.")
    by_time = {_timestamp_from_key(obj.key): obj for obj in objects}
    issue = max(by_time)
    expected = [issue - timedelta(minutes=CADENCE_MINUTES * i) for i in range(9, -1, -1)]
    return [(stamp, by_time.get(stamp)) for stamp in expected]


class _FileLock:
    def __init__(self, path: Path, timeout: float = 120) -> None:
        self.path, self.timeout, self.fd = path, timeout, None

    def __enter__(self):
        started = time.monotonic()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                if time.monotonic() - started > self.timeout:
                    raise TimeoutError(f"timed out waiting for cache lock: {self.path}")
                time.sleep(0.1)

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def download_atomic(obj: S3Object, cache_dir: Path = DEFAULT_CACHE,
                    get: Callable[..., requests.Response] = requests.get) -> Path:
    raw = cache_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    destination = raw / Path(obj.key).name
    if destination.is_file() and destination.stat().st_size == obj.size:
        return destination
    lock = destination.with_suffix(destination.suffix + ".lock")
    with _FileLock(lock):
        if destination.is_file() and destination.stat().st_size == obj.size:
            return destination
        response = get(object_url(BUCKET, obj.key), stream=True, timeout=120)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=raw, prefix="mrms-", suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
        try:
            if temporary.stat().st_size != obj.size:
                raise LiveDataUnavailable(f"incomplete MRMS download for {obj.key}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def forecast_cache_identity(tile: TileDefinition, issue_time: datetime,
                            model_version: str, config_sha256: str) -> str:
    raw = json.dumps({"tile": tile.identity, "issue": issue_time.astimezone(UTC).isoformat(),
                      "model": model_version, "config": config_sha256}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def prepare_live_history(latitude: float, longitude: float, *, cache_dir: Path = DEFAULT_CACHE,
                         now: datetime | None = None,
                         listing: Callable[[datetime], list[S3Object]] = daily_objects,
                         downloader: Callable[[S3Object, Path], Path] | None = None,
                         decoder: Callable[[Path, list[float]], tuple[np.ndarray, np.ndarray, np.ndarray]] = decode_mrms_crop,
                         ) -> LiveRadarHistory:
    tile = tile_for_location(latitude, longitude)
    selected = regular_history(latest_radar_objects(now, listing))
    if sum(obj is not None for _, obj in selected) < MIN_AVAILABLE_FRAMES:
        raise LiveDataUnavailable("Recent radar history has too many missing scans; please try again shortly.")
    download = downloader or (lambda obj, root: download_atomic(obj, root))
    bbox = [float(tile.longitude.min()) - 0.05, float(tile.latitude.min()) - 0.05,
            float(tile.longitude.max()) + 0.05, float(tile.latitude.max()) + 0.05]
    frames, valid, keys, missing = [], [], [], []
    for stamp, obj in selected:
        if obj is None:
            frames.append(np.zeros((128, 128), np.float32)); valid.append(np.zeros((128, 128), bool))
            keys.append(""); missing.append(stamp); continue
        source = download(obj, cache_dir)
        values, source_lat, source_lon = decoder(source, bbox)
        field = _interpolate_regular_latlon(values, source_lat, source_lon,
                                            tile.latitude, tile.longitude)
        mask = np.isfinite(field)
        if mask.mean() < 0.90:
            raise LiveDataUnavailable(f"Insufficient radar coverage at {stamp.isoformat()}.")
        frames.append(np.where(mask, np.maximum(field, 0), 0).astype(np.float32))
        valid.append(mask); keys.append(obj.key)
    issue = selected[-1][0]
    return LiveRadarHistory(np.stack(frames), tuple(t for t, _ in selected), np.stack(valid), tile,
                            issue, tuple(keys), tuple(missing))


def geocode_location(query: str, *, get: Callable[..., requests.Response] = requests.get) -> dict[str, object]:
    if not query.strip():
        raise ValueError("Enter a city, address, or place.")
    response = get("https://nominatim.openstreetmap.org/search",
                   params={"q": query, "format": "jsonv2", "limit": 5, "countrycodes": "ca"},
                   headers={"User-Agent": "OntarioNowcast/1.0 (research application)"}, timeout=15)
    response.raise_for_status()
    results = response.json()
    if not results:
        raise LookupError("No matching location was found.")
    candidates = [{"name": item["display_name"], "latitude": float(item["lat"]),
                   "longitude": float(item["lon"])} for item in results]
    return {"candidates": candidates}


def live_smoke_test(latitude: float = 43.6532, longitude: float = -79.3832) -> LiveRadarHistory:
    """Optional network integration test; downloads current NOAA data."""
    return prepare_live_history(latitude, longitude)
