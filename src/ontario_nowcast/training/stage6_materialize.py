"""Materialize immutable A+ and issue-safe HRRR inputs for Stage 6."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..models.radar_convlstm import build_radar_model, expected_rate_mm_hr
from ..preprocessing.grid import transform_coordinates
from .stage4c_baselines import _decode_apcp, _download
from .stage4c_tensors import _apply, _decode, _weights
from .stage6_prepare import EXPECTED_A_PLUS, EXPECTED_HOLDOUT, sha

ROOT = Path("artifacts/stage_6")


def _product(model_issue: pd.Timestamp, forecast_hour: int) -> str:
    return (
        f"hrrr.{model_issue:%Y%m%d}/conus/hrrr.t{model_issue:%H}z.wrfsfcf{forecast_hour:02d}.grib2"
    )


def run() -> dict:
    procedure_path = ROOT / "frozen_procedure_manifest.json"
    gate = json.loads((ROOT / "preprediction_identity_gate.json").read_text())
    if not all(gate[key] for key in ("a_plus_hash_verified", "holdout_hash_verified")):
        raise RuntimeError("preprediction identity gate is not frozen")
    if (
        sha(Path("artifacts/stage_4c/frozen_holdout/stage4c_final_holdout_manifest.csv"))
        != EXPECTED_HOLDOUT
    ):
        raise RuntimeError("holdout manifest changed")
    if sha(Path("artifacts/stage_4b/a_plus_radar/model_best.pt")) != EXPECTED_A_PLUS:
        raise RuntimeError("A+ checkpoint changed")
    manifest = pd.read_csv("artifacts/stage_4c/frozen_holdout/exact_tensor_manifest.csv")
    if len(manifest) != 68:
        raise RuntimeError("exact 68-row tensor manifest required")
    products_by_identity, products = {}, set()
    for row in manifest.itertuples(index=False):
        payload = np.load(row.tensor_path)
        records = json.loads(str(payload["hrrr_provenance_json"]))
        issues = {pd.Timestamp(record["model_issue_time_utc"]) for record in records}
        if len(issues) != 1:
            raise RuntimeError(f"{row.identity}: inconsistent HRRR model issue")
        model_issue = issues.pop()
        pair = [_product(model_issue, hour) for hour in (2, 3)]
        products_by_identity[row.identity] = (model_issue, pair)
        products.update(pair)
    source_dir = ROOT / "hrrr_apcp_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for product in sorted(products):
            destination = source_dir / product.replace("/conus/", "__").replace("/", "__").replace(
                ".grib2", ".apcp.grib2"
            )
            futures[pool.submit(_download, product, destination, ROOT / "apcp_downloads.jsonl")] = (
                product
            )
        for number, future in enumerate(as_completed(futures), 1):
            source_rows.append(future.result())
            if number == 1 or number % 20 == 0:
                print(f"[stage6-apcp] {number}/{len(futures)}", flush=True)
    for row in source_rows:
        values, units, step_range = _decode_apcp(Path(row["local_path"]))
        row.update(
            units=units,
            step_range=step_range,
            finite_fraction=float(np.isfinite(values).mean()),
            minimum=float(np.nanmin(values)),
            maximum=float(np.nanmax(values)),
        )
    sources = pd.DataFrame(source_rows).sort_values("product_key")
    sources.to_csv(ROOT / "hrrr_apcp_source_manifest.csv", index=False)
    source_lookup = sources.set_index("product_key")
    geometry_source = pd.read_csv(
        "artifacts/stage_4c/holdout_sources/stage4_hrrr_materialized_sources.csv"
    ).iloc[0]
    _, geometry = _decode(Path(geometry_source.local_path), include_geometry=True)
    assert geometry is not None
    sx, sy = geometry
    config = yaml.safe_load(
        Path("configs/experiments/stage_4b_a_plus_radar_capacity.yaml").read_text()
    )
    metadata = json.loads(
        Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text()
    )
    norm = metadata["normalization"]["radar"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_radar_model({"model": config["model"], "data": config["data"]}).to(device)
    model.load_state_dict(
        torch.load("artifacts/stage_4b/a_plus_radar/model_best.pt", map_location=device)
    )
    model.eval()
    rows, weight_cache = [], {}
    cache_dir = ROOT / "cache"
    for number, row in enumerate(manifest.itertuples(index=False), 1):
        payload = np.load(row.tensor_path)
        radar = payload["radar_rate_mm_hr"]
        mask = payload["radar_valid_mask"].astype(np.float32)
        radar_input = np.stack(
            [(np.log1p(np.nan_to_num(radar)) - norm["mean"]) / norm["std"], mask], 1
        ).astype(np.float32)
        with torch.no_grad():
            logits, intensity = model(torch.from_numpy(radar_input[None]).to(device))
            expected = expected_rate_mm_hr(logits, intensity)
        tile = tuple(int(value) for value in payload["tile_indices"])
        if tile not in weight_cache:
            tx, ty = transform_coordinates(
                payload["longitude"],
                payload["latitude"],
                source_crs="EPSG:4326",
                destination_crs="EPSG:3978",
            )
            weight_cache[tile] = _weights(sx, sy, tx, ty)
        vertices, weights = weight_cache[tile]
        model_issue, pair = products_by_identity[row.identity]
        hourly, provenance = [], []
        for product in pair:
            source = source_lookup.loc[product]
            values, units, step_range = _decode_apcp(Path(source.local_path))
            hourly.append(_apply(values, vertices, weights, radar.shape[1:]))
            forecast_hour = 2 if "f02" in product else 3
            provenance.append(
                {
                    "product_key": product,
                    "sha256": source.sha256,
                    "model_issue_time_utc": model_issue.isoformat(),
                    "forecast_hour": forecast_hour,
                    "valid_interval": step_range,
                    "units": units,
                }
            )
        destination = cache_dir / f"{row.identity}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            aplus_logits=logits[0].cpu().numpy().astype(np.float32),
            aplus_expected_rate_mm_hr=expected[0].cpu().numpy().astype(np.float32),
            hrrr_apcp_hourly=np.stack(hourly).astype(np.float32),
            hrrr_provenance_json=np.asarray(json.dumps(provenance)),
            source_tensor_sha256=np.asarray(row.sha256),
            procedure_sha256=np.asarray(sha(procedure_path)),
        )
        rows.append(
            {
                **row._asdict(),
                "stage6_cache_path": destination.as_posix(),
                "stage6_cache_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "hrrr_model_issue_time_utc": model_issue.isoformat(),
                "hrrr_f02_product": pair[0],
                "hrrr_f03_product": pair[1],
                "future_radar_in_inputs": False,
                "later_hrrr_cycle_used": False,
            }
        )
        print(f"[stage6-cache] {number}/{len(manifest)}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "stage6_cache_manifest.csv", index=False)
    first = table.iloc[0]
    first_payload = np.load(first.tensor_path)
    first_cache = np.load(first.stage6_cache_path)
    with torch.no_grad():
        repeated_logits, _ = model(
            torch.from_numpy(
                np.stack(
                    [
                        (np.log1p(np.nan_to_num(first_payload["radar_rate_mm_hr"])) - norm["mean"])
                        / norm["std"],
                        first_payload["radar_valid_mask"].astype(np.float32),
                    ],
                    1,
                )[None].astype(np.float32)
            ).to(device)
        )
    identity = bool(np.array_equal(repeated_logits[0].cpu().numpy(), first_cache["aplus_logits"]))
    integrity = {
        "rows": len(table),
        "source_products": len(sources),
        "all_tensors_present": bool(table.stage6_cache_path.map(lambda x: Path(x).exists()).all()),
        "a_plus_repeat_bit_exact": identity,
        "a_plus_checkpoint_sha256": sha(Path("artifacts/stage_4b/a_plus_radar/model_best.pt")),
        "holdout_manifest_sha256": sha(
            Path("artifacts/stage_4c/frozen_holdout/stage4c_final_holdout_manifest.csv")
        ),
        "procedure_sha256": sha(procedure_path),
        "future_radar_in_inputs": False,
        "later_hrrr_cycle_used": False,
        "normalization_refit": False,
        "calibration_fit": False,
        "scoreable": bool(len(table) == 68 and identity),
    }
    (ROOT / "materialization_integrity.json").write_text(json.dumps(integrity, indent=2))
    return integrity


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
