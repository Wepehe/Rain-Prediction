"""Build a small, physically georeferenced multimodal tensor on EPSG:3978."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator

from ..data.manifest import sha256_file
from ..data.multimodal import _goes_scan_time
from .grid import transform_coordinates
from .units import kilometres_per_hour_to_metres_per_second


def analysis_grid(
    bbox_wgs84: tuple[float, float, float, float], resolution_metres: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    west, south, east, north = bbox_wgs84
    corner_lon = np.array([west, west, east, east])
    corner_lat = np.array([south, north, south, north])
    corner_x, corner_y = transform_coordinates(
        corner_lon, corner_lat, source_crs="EPSG:4326", destination_crs="EPSG:3978"
    )
    x = np.arange(np.floor(corner_x.min() / resolution_metres) * resolution_metres,
                  np.ceil(corner_x.max() / resolution_metres) * resolution_metres + 1,
                  resolution_metres)
    y = np.arange(np.floor(corner_y.min() / resolution_metres) * resolution_metres,
                  np.ceil(corner_y.max() / resolution_metres) * resolution_metres + 1,
                  resolution_metres)
    grid_x, grid_y = np.meshgrid(x, y)
    longitude, latitude = transform_coordinates(
        grid_x, grid_y, source_crs="EPSG:3978", destination_crs="EPSG:4326"
    )
    return x, y, longitude, latitude


def _interpolate_regular_latlon(
    values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:
    if latitude[0] > latitude[-1]:
        latitude = latitude[::-1]
        values = values[::-1]
    if longitude[0] > longitude[-1]:
        longitude = longitude[::-1]
        values = values[:, ::-1]
    interpolator = RegularGridInterpolator(
        (latitude, longitude), values, bounds_error=False, fill_value=np.nan
    )
    points = np.column_stack((target_latitude.ravel(), target_longitude.ravel()))
    return interpolator(points).reshape(target_latitude.shape).astype(np.float32)


def _interpolate_goes(path: Path, target_lon: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
    import xarray as xr
    from pyproj import CRS

    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        projection = dataset.goes_imager_projection.attrs
        height = float(projection["perspective_point_height"])
        geos = CRS.from_proj4(
            "+proj=geos "
            f"+h={height} +lon_0={float(projection['longitude_of_projection_origin'])} "
            f"+a={float(projection['semi_major_axis'])} +b={float(projection['semi_minor_axis'])} "
            f"+sweep={projection['sweep_angle_axis']} +units=m +no_defs"
        )
        target_x, target_y = transform_coordinates(
            target_lon, target_lat, source_crs="EPSG:4326", destination_crs=geos.to_string()
        )
        source_x = dataset.x.values * height
        source_y = dataset.y.values * height
        values = dataset.CMI.values.astype(np.float32)
        if source_y[0] > source_y[-1]:
            source_y = source_y[::-1]
            values = values[::-1]
        interpolator = RegularGridInterpolator(
            (source_y, source_x), values, bounds_error=False, fill_value=np.nan
        )
        points = np.column_stack((target_y.ravel(), target_x.ravel()))
        return interpolator(points).reshape(target_lon.shape).astype(np.float32)


def _hrrr_fields(path: Path, target_x: np.ndarray, target_y: np.ndarray) -> dict[str, np.ndarray]:
    import cfgrib

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")
        groups = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    variables = {name: group[name] for group in groups for name in group.data_vars}
    reference = next(iter(variables.values()))
    longitude = ((reference.longitude.values + 180) % 360) - 180
    latitude = reference.latitude.values
    source_x, source_y = transform_coordinates(
        longitude, latitude, source_crs="EPSG:4326", destination_crs="EPSG:3978"
    )
    padding = 100_000
    keep = (
        (source_x >= target_x.min() - padding)
        & (source_x <= target_x.max() + padding)
        & (source_y >= target_y.min() - padding)
        & (source_y <= target_y.max() + padding)
    )
    source_points = np.column_stack((source_x[keep], source_y[keep]))
    target_points = np.column_stack((target_x.ravel(), target_y.ravel()))
    result = {}
    for name, variable in variables.items():
        source_values = variable.values[keep]
        finite = np.isfinite(source_values)
        if np.count_nonzero(finite) < 3:
            result[name] = np.full(target_x.shape, np.nan, dtype=np.float32)
            continue
        linear = LinearNDInterpolator(source_points[finite], source_values[finite], fill_value=np.nan)
        interpolated = linear(target_points)
        missing = ~np.isfinite(interpolated)
        if np.any(missing):
            nearest = NearestNDInterpolator(source_points[finite], source_values[finite])
            interpolated[missing] = nearest(target_points[missing])
        result[name] = interpolated.reshape(target_x.shape).astype(np.float32)
    return result


def _station_fields(
    paths: list[Path], valid_time: pd.Timestamp, target_x: np.ndarray, target_y: np.ndarray
) -> dict[str, np.ndarray]:
    rows = []
    fixed_standard_time = timezone(timedelta(hours=-5))
    for path in paths:
        frame = pd.read_csv(path)
        frame["time_utc"] = (
            pd.to_datetime(frame["Date/Time (LST)"], errors="coerce")
            .dt.tz_localize(fixed_standard_time)
            .dt.tz_convert("UTC")
        )
        nearest = (frame["time_utc"] - valid_time).abs().idxmin()
        rows.append(frame.loc[nearest])
    station_x, station_y = transform_coordinates(
        np.array([float(row["Longitude (x)"]) for row in rows]),
        np.array([float(row["Latitude (y)"]) for row in rows]),
        source_crs="EPSG:4326",
        destination_crs="EPSG:3978",
    )
    distance = np.sqrt(
        (target_x[..., None] - station_x) ** 2 + (target_y[..., None] - station_y) ** 2
    )
    nearest_distance = distance.min(axis=-1)
    weights = 1 / np.maximum(distance, 1_000) ** 2
    weights /= weights.sum(axis=-1, keepdims=True)
    columns = {
        "station_temperature_c": "Temp (°C)",
        "station_dewpoint_c": "Dew Point Temp (°C)",
        "station_wind_m_s": "Wind Spd (km/h)",
        "station_pressure_kpa": "Stn Press (kPa)",
    }
    result = {"station_distance_km": (nearest_distance / 1000).astype(np.float32)}
    for output_name, column in columns.items():
        source = pd.to_numeric(pd.Series([row[column] for row in rows]), errors="coerce").to_numpy()
        if output_name == "station_wind_m_s":
            source = kilometres_per_hour_to_metres_per_second(source)
        field = np.sum(weights * source, axis=-1)
        field[nearest_distance > 150_000] = np.nan
        result[output_name] = field.astype(np.float32)
    return result


def build_fused_sample(
    radar_sample: Path,
    data_root: Path,
    output: Path,
    bbox: tuple[float, float, float, float] = (-81.0, 42.0, -77.5, 45.0),
) -> Path:
    radar = np.load(radar_sample)
    radar_times = pd.to_datetime(radar["times"], utc=True)
    target_times = pd.date_range(radar_times[0], radar_times[-1], freq="1h")
    x, y, longitude, latitude = analysis_grid(bbox, 2_000)
    target_x, target_y = np.meshgrid(x, y)

    goes_paths = sorted((data_root / "raw" / "satellite" / "goes16").glob("*.nc"))
    goes_by_hour = {
        pd.Timestamp(_goes_scan_time(path.name)).floor("h"): path for path in goes_paths
    }
    hrrr_paths = sorted((data_root / "raw" / "nwp" / "hrrr").glob("*.grib2"))
    hrrr_by_time = {}
    for path in hrrr_paths:
        match = re.search(r"hrrr\.(\d{10})\.subset", path.name)
        if match:
            valid_time = pd.to_datetime(match.group(1), format="%Y%m%d%H", utc=True)
            hrrr_by_time[valid_time] = path
    station_paths = sorted((data_root / "raw" / "stations" / "eccc").glob("*.csv"))

    output_fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "radar_rate_mm_hr",
            "goes_c13_k",
            "nwp_u10_m_s",
            "nwp_v10_m_s",
            "nwp_t2m_k",
            "nwp_d2m_k",
            "nwp_mslma_pa",
            "nwp_cape_j_kg",
            "station_temperature_c",
            "station_dewpoint_c",
            "station_wind_m_s",
            "station_pressure_kpa",
            "station_distance_km",
        )
    }
    hrrr_names = {
        "u10": "nwp_u10_m_s",
        "v10": "nwp_v10_m_s",
        "t2m": "nwp_t2m_k",
        "d2m": "nwp_d2m_k",
        "mslma": "nwp_mslma_pa",
        "cape": "nwp_cape_j_kg",
    }
    for valid_time in target_times:
        radar_index = int(np.argmin(np.abs(radar_times - valid_time)))
        output_fields["radar_rate_mm_hr"].append(
            _interpolate_regular_latlon(
                radar["rate_mm_hr"][radar_index],
                radar["latitude"],
                radar["longitude"],
                latitude,
                longitude,
            )
        )
        output_fields["goes_c13_k"].append(
            _interpolate_goes(goes_by_hour[valid_time.floor("h")], longitude, latitude)
        )
        nwp = _hrrr_fields(hrrr_by_time[valid_time], target_x, target_y)
        for source_name, output_name in hrrr_names.items():
            output_fields[output_name].append(nwp[source_name])
        stations = _station_fields(station_paths, valid_time, target_x, target_y)
        for name, values in stations.items():
            output_fields[name].append(values)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        times=np.asarray([time.isoformat() for time in target_times]),
        x_m=x,
        y_m=y,
        longitude=longitude.astype(np.float32),
        latitude=latitude.astype(np.float32),
        crs=np.asarray("EPSG:3978"),
        **{name: np.stack(values) for name, values in output_fields.items()},
    )
    record = {
        "output": output.as_posix(),
        "sha256": sha256_file(output),
        "shape": [len(target_times), len(y), len(x)],
        "crs": "EPSG:3978",
        "resolution_metres": 2000,
        "bbox_wgs84": list(bbox),
        "times": [time.isoformat() for time in target_times],
        "radar_interpolation": "bilinear from native MRMS lat/lon",
        "satellite_interpolation": "bilinear in GOES fixed-grid projection",
        "nwp_interpolation": "linear interpolation in EPSG:3978 with nearest fill outside the local convex hull",
        "station_representation": "two-station inverse-distance-squared; masked beyond 150 km",
    }
    manifest = data_root / "metadata" / "fused_builds.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def plot_fused_sample(path: Path, destination: Path, time_index: int = 3) -> None:
    import matplotlib.pyplot as plt

    payload = np.load(path)
    fields = (
        ("radar_rate_mm_hr", "Radar rate [mm h⁻¹]", "turbo", 0, 25),
        ("goes_c13_k", "GOES C13 brightness temperature [K]", "gray_r", 210, 300),
        ("nwp_cape_j_kg", "HRRR surface CAPE [J kg⁻¹]", "magma", 0, 3000),
        ("nwp_t2m_k", "HRRR 2 m temperature [K]", "coolwarm", 288, 306),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis, (name, title, colour_map, minimum, maximum) in zip(
        axes.flat, fields, strict=True
    ):
        image = axis.imshow(
            payload[name][time_index], origin="lower", cmap=colour_map, vmin=minimum, vmax=maximum
        )
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(
        f"Synchronized EPSG:3978 sample — "
        f"{pd.Timestamp(str(payload['times'][time_index])).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("radar_sample", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/fused/toronto_2024_07_16.npz")
    )
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = build_fused_sample(args.radar_sample, args.data_root, args.output)
    if args.plot is not None:
        plot_fused_sample(result, args.plot)
    print(result)


if __name__ == "__main__":
    main()
