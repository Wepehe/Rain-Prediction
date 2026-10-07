"""Materialize Stage 4C thermodynamic fields on Stage 4 radar tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import eccodes
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

from ..preprocessing.grid import transform_coordinates

VARIABLES = ("TMP_2m", "DPT_2m", "CAPE_surface", "CIN_surface", "PWAT_column")
SHORT_NAMES = {"TMP_2m": "2t", "DPT_2m": "2d", "CAPE_surface": "cape", "CIN_surface": "cin", "PWAT_column": "pwat"}


def _decode(path: Path, *, include_geometry: bool = False) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    fields: dict[str, np.ndarray] = {}
    geometry = None
    with path.open("rb") as handle:
        while True:
            message = eccodes.codes_grib_new_from_file(handle)
            if message is None:
                break
            try:
                short = str(eccodes.codes_get(message, "shortName"))
                if short in SHORT_NAMES.values():
                    fields[short] = np.asarray(eccodes.codes_get_values(message), dtype=np.float32)
                    if include_geometry and geometry is None:
                        lat = np.asarray(eccodes.codes_get_array(message, "latitudes"))
                        lon = np.asarray(eccodes.codes_get_array(message, "longitudes"))
                        lon = (lon + 180) % 360 - 180
                        geometry = transform_coordinates(lon, lat, source_crs="EPSG:4326", destination_crs="EPSG:3978")
            finally:
                eccodes.codes_release(message)
    missing = set(SHORT_NAMES.values()) - set(fields)
    if missing:
        raise RuntimeError(f"{path} missing HRRR fields {sorted(missing)}")
    return fields, geometry


def _weights(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray, target_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padding = 100_000.0
    keep = (source_x >= target_x.min()-padding) & (source_x <= target_x.max()+padding) & (source_y >= target_y.min()-padding) & (source_y <= target_y.max()+padding)
    global_indices = np.flatnonzero(keep)
    source = np.column_stack((source_x[keep], source_y[keep]))
    target = np.column_stack((target_x.ravel(), target_y.ravel()))
    triangulation = Delaunay(source)
    simplex = triangulation.find_simplex(target)
    vertices = np.empty((len(target), 3), dtype=np.int64)
    weights = np.empty((len(target), 3), dtype=np.float64)
    inside = simplex >= 0
    transform = triangulation.transform[simplex[inside]]
    delta = target[inside] - transform[:, 2]
    bary = np.einsum("nij,nj->ni", transform[:, :2], delta)
    weights[inside, :2] = bary
    weights[inside, 2] = 1-bary.sum(axis=1)
    vertices[inside] = triangulation.simplices[simplex[inside]]
    if (~inside).any():
        _, nearest = cKDTree(source).query(target[~inside])
        vertices[~inside] = np.column_stack((nearest, nearest, nearest))
        weights[~inside] = (1.0, 0.0, 0.0)
    return global_indices[vertices], weights.astype(np.float32)


def _apply(values: np.ndarray, vertices: np.ndarray, weights: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    sampled = np.sum(values[vertices] * weights, axis=1)
    return sampled.reshape(shape).astype(np.float32)


def materialize(output_dir: Path) -> dict[str, Any]:
    radar_manifest = pd.read_csv("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv")
    required = pd.read_csv("artifacts/stage_4c/materialization/stage4_hrrr_required_sources.csv")
    sources = pd.read_csv("artifacts/stage_4c/materialization/stage4_hrrr_materialized_sources.csv").set_index("product_key")
    first_path = Path(sources.iloc[0].local_path)
    _, geometry = _decode(first_path, include_geometry=True)
    assert geometry is not None
    source_x, source_y = geometry
    tile_weights: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    rows, failures = [], []
    output_root = output_dir / "thermodynamic_tensors"
    for index, row in enumerate(radar_manifest.itertuples(index=False), 1):
        try:
            radar_payload = np.load(row.tensor_path)
            lon, lat = radar_payload["longitude"], radar_payload["latitude"]
            target_x, target_y = transform_coordinates(lon, lat, source_crs="EPSG:4326", destination_crs="EPSG:3978")
            tile_key = (int(row.tile_y_min), int(row.tile_y_max), int(row.tile_x_min), int(row.tile_x_max))
            if tile_key not in tile_weights:
                tile_weights[tile_key] = _weights(source_x, source_y, target_x, target_y)
            vertices, interpolation_weights = tile_weights[tile_key]
            anchor = required[required.anchor_id.astype(str).eq(str(row.anchor_id))]
            valid_times = sorted(anchor.model_valid_time_utc.unique())
            if len(valid_times) != 3:
                raise ValueError(f"expected three HRRR valid times, found {len(valid_times)}")
            frames, source_records = [], []
            for valid_time in valid_times:
                subset = anchor[anchor.model_valid_time_utc.eq(valid_time)]
                product_keys = subset.product_key.unique()
                if len(product_keys) != 1:
                    raise ValueError("C1 variables do not share one HRRR surface product")
                product_key = str(product_keys[0])
                source_path = Path(sources.loc[product_key].local_path)
                decoded, _ = _decode(source_path)
                frame = np.stack([_apply(decoded[SHORT_NAMES[name]], vertices, interpolation_weights, lon.shape) for name in VARIABLES])
                frames.append(frame)
                source_records.append({"model_issue_time_utc": str(subset.model_issue_time_utc.iloc[0]), "model_valid_time_utc": str(valid_time), "forecast_hour": int(subset.forecast_hour.iloc[0]), "product_key": product_key})
            tensor = np.stack(frames).astype(np.float32)
            destination = output_root / str(row.split) / f"{row.anchor_id}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, thermodynamics=tensor, thermodynamics_valid_mask=np.isfinite(tensor).astype(np.float32), variable_names=np.asarray(VARIABLES), source_records_json=np.asarray(json.dumps(source_records)), interpolation=np.asarray("linear in EPSG:3978; nearest only outside triangulation"))
            rows.append({"anchor_id": row.anchor_id, "event_id": row.event_id, "split": row.split, "event_class": row.event_class, "radar_tensor_path": row.tensor_path, "thermodynamic_tensor_path": destination.as_posix(), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "shape": "x".join(map(str,tensor.shape)), "valid_fraction": float(np.isfinite(tensor).mean()), "model_issue_times": json.dumps([x["model_issue_time_utc"] for x in source_records]), "valid_times": json.dumps(valid_times), "forecast_hours": json.dumps([x["forecast_hour"] for x in source_records])})
            print(f"[stage4c-tensors] {index}/{len(radar_manifest)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"anchor_id": row.anchor_id, "error": str(exc)})
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "stage4c_thermodynamic_tensor_manifest.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "stage4c_thermodynamic_tensor_failures.csv", index=False)
    summary = {"anchors_requested": len(radar_manifest), "anchors_materialized": len(table), "failures": len(failures), "variables": list(VARIABLES), "shape": "3x5x128x128", "complete": bool(len(table)==len(radar_manifest) and not failures), "stage_c_training_started": False}
    (output_dir / "stage4c_thermodynamic_tensor_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=Path("artifacts/stage_4c/tensors"))
    args=parser.parse_args()
    print(json.dumps(materialize(args.output_dir),indent=2))


if __name__ == "__main__":
    main()
