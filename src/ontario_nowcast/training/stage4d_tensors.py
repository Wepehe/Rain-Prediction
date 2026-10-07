"""Remap Stage 4D winds and derive projection-aware dynamical fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import eccodes
import numpy as np
import pandas as pd

from ..preprocessing.grid import transform_coordinates
from .stage4c_tensors import _apply, _weights

NATIVE_KEYS = {
    ("10u", "heightAboveGround", 10): "UGRD_10m",
    ("10v", "heightAboveGround", 10): "VGRD_10m",
    ("u", "isobaricInhPa", 850): "UGRD_850hPa",
    ("v", "isobaricInhPa", 850): "VGRD_850hPa",
    ("u", "isobaricInhPa", 500): "UGRD_500hPa",
    ("v", "isobaricInhPa", 500): "VGRD_500hPa",
    ("w", "isobaricInhPa", 700): "VVEL_700hPa",
}
VARIABLES = (
    "UGRD_10m",
    "VGRD_10m",
    "UGRD_850hPa",
    "VGRD_850hPa",
    "VVEL_700hPa",
    "CONVERGENCE_850hPa",
    "VORTICITY_850hPa",
    "SHEAR_U_850_500hPa",
    "SHEAR_V_850_500hPa",
    "SHEAR_MAG_850_500hPa",
)
ALLOWED_SPLITS = {"train", "dev", "stage4_dev_repair"}


def _decode(path: Path, geometry: bool = False):
    fields: dict[str, np.ndarray] = {}
    coordinates = None
    with path.open("rb") as handle:
        while True:
            message = eccodes.codes_grib_new_from_file(handle)
            if message is None:
                break
            try:
                key = (
                    str(eccodes.codes_get(message, "shortName")),
                    str(eccodes.codes_get(message, "typeOfLevel")),
                    int(eccodes.codes_get(message, "level")),
                )
                if key in NATIVE_KEYS:
                    fields[NATIVE_KEYS[key]] = np.asarray(
                        eccodes.codes_get_values(message), dtype=np.float32
                    )
                    if geometry and coordinates is None:
                        lat = np.asarray(eccodes.codes_get_array(message, "latitudes"))
                        lon = np.asarray(eccodes.codes_get_array(message, "longitudes"))
                        lon = (lon + 180) % 360 - 180
                        coordinates = transform_coordinates(
                            lon, lat, source_crs="EPSG:4326", destination_crs="EPSG:3978"
                        )
            finally:
                eccodes.codes_release(message)
    return fields, coordinates


def _derivative(field: np.ndarray, coordinate: np.ndarray, axis: int) -> np.ndarray:
    spacing = np.gradient(coordinate, axis=axis)
    return np.divide(
        np.gradient(field, axis=axis),
        spacing,
        out=np.full_like(field, np.nan, dtype=np.float32),
        where=np.abs(spacing) > 1,
    ).astype(np.float32)


def _derive(native: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u850, v850 = native["UGRD_850hPa"], native["VGRD_850hPa"]
    du_dx, dv_dy = _derivative(u850, x, 1), _derivative(v850, y, 0)
    dv_dx, du_dy = _derivative(v850, x, 1), _derivative(u850, y, 0)
    shear_u = native["UGRD_500hPa"] - u850
    shear_v = native["VGRD_500hPa"] - v850
    values = {
        **native,
        "CONVERGENCE_850hPa": -(du_dx + dv_dy),
        "VORTICITY_850hPa": dv_dx - du_dy,
        "SHEAR_U_850_500hPa": shear_u,
        "SHEAR_V_850_500hPa": shear_v,
        "SHEAR_MAG_850_500hPa": np.hypot(shear_u, shear_v),
    }
    return np.stack([values[name] for name in VARIABLES]).astype(np.float32)


def run(output_dir: Path = Path("artifacts/stage_4d/tensors")) -> dict[str, Any]:
    radar = pd.read_csv("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv")
    radar = radar[radar.split.isin(ALLOWED_SPLITS)].reset_index(drop=True)
    required = pd.read_csv("artifacts/stage_4d/source_gate/stage4d_hrrr_required_sources.csv")
    sources = pd.read_csv(
        "artifacts/stage_4d/source_gate/stage4d_hrrr_materialized_sources.csv"
    ).set_index("product_key")
    _, geometry = _decode(Path(sources.iloc[0].local_path), geometry=True)
    if geometry is None:
        raise RuntimeError("HRRR geometry could not be decoded")
    source_x, source_y = geometry
    weight_cache = {}
    rows, failures = [], []
    for number, row in enumerate(radar.itertuples(index=False), 1):
        try:
            radar_payload = np.load(row.tensor_path)
            lon, lat = radar_payload["longitude"], radar_payload["latitude"]
            x, y = transform_coordinates(
                lon, lat, source_crs="EPSG:4326", destination_crs="EPSG:3978"
            )
            tile = (row.tile_y_min, row.tile_y_max, row.tile_x_min, row.tile_x_max)
            if tile not in weight_cache:
                weight_cache[tile] = _weights(source_x, source_y, x, y)
            vertices, weights = weight_cache[tile]
            anchor = required[required.anchor_id.astype(str).eq(str(row.anchor_id))]
            valid_times = sorted(anchor.model_valid_time_utc.unique())
            if len(valid_times) != 3:
                raise ValueError(f"expected three valid times, found {len(valid_times)}")
            frames, records = [], []
            for valid_time in valid_times:
                subset = anchor[anchor.model_valid_time_utc.eq(valid_time)]
                native: dict[str, np.ndarray] = {}
                product_keys = sorted(subset.product_key.unique())
                for product_key in product_keys:
                    decoded, _ = _decode(Path(sources.loc[product_key].local_path))
                    native.update(
                        {
                            name: _apply(values, vertices, weights, lon.shape)
                            for name, values in decoded.items()
                        }
                    )
                missing = set(NATIVE_KEYS.values()) - set(native)
                if missing:
                    raise RuntimeError(f"missing native fields {sorted(missing)}")
                frames.append(_derive(native, x, y))
                records.append(
                    {
                        "model_issue_time_utc": str(subset.model_issue_time_utc.iloc[0]),
                        "model_valid_time_utc": str(valid_time),
                        "forecast_hour": int(subset.forecast_hour.iloc[0]),
                        "product_keys": product_keys,
                    }
                )
            tensor = np.stack(frames)
            destination = output_dir / "dynamics_tensors" / row.split / f"{row.anchor_id}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                dynamics=tensor,
                dynamics_valid_mask=np.isfinite(tensor).astype(np.float32),
                variable_names=np.asarray(VARIABLES),
                source_records_json=np.asarray(json.dumps(records)),
                derivative_definition=np.asarray("EPSG:3978 local coordinate gradients"),
            )
            rows.append(
                {
                    "anchor_id": row.anchor_id,
                    "event_id": row.event_id,
                    "split": row.split,
                    "event_class": row.event_class,
                    "radar_tensor_path": row.tensor_path,
                    "dynamics_tensor_path": destination.as_posix(),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "shape": "x".join(map(str, tensor.shape)),
                    "valid_fraction": float(np.isfinite(tensor).mean()),
                    "model_issue_times": json.dumps(
                        [item["model_issue_time_utc"] for item in records]
                    ),
                    "valid_times": json.dumps(valid_times),
                    "forecast_hours": json.dumps([item["forecast_hour"] for item in records]),
                }
            )
            print(f"[stage4d-tensors] {number}/{len(radar)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"anchor_id": row.anchor_id, "error": str(exc)})
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "stage4d_dynamics_tensor_manifest.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "stage4d_dynamics_tensor_failures.csv", index=False)
    summary = {
        "anchors_requested": len(radar),
        "anchors_materialized": len(table),
        "failures": len(failures),
        "variables": list(VARIABLES),
        "shape": "3x10x128x128",
        "complete": len(table) == len(radar) and not failures,
        "protected_2023_holdout_used": False,
        "training_started": False,
    }
    (output_dir / "stage4d_tensor_gate.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
