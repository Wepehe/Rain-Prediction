"""Cache detached frozen-A+ outputs and honest hourly HRRR APCP for H1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..models.radar_convlstm import build_radar_model, expected_rate_mm_hr
from ..preprocessing.grid import transform_coordinates
from .stage4c_baselines import _decode_apcp
from .stage4c_tensors import _apply, _decode, _weights

ALLOWED_SPLITS = {"train", "dev", "stage4_dev_repair"}
EXPECTED_CHECKPOINT = "63e049c5ea75c3aa6af7b461fe6e9bca2db3c51198a667e71f02a7c6217efe70"


def run(output_dir: Path = Path("artifacts/stage_5/cache")) -> dict[str, object]:
    checkpoint = Path("artifacts/stage_4b/a_plus_radar/model_best.pt")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise RuntimeError("frozen A+ checkpoint hash changed")
    radar_manifest = pd.read_csv(
        "artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"
    )
    radar_manifest = radar_manifest[radar_manifest.split.isin(ALLOWED_SPLITS)].reset_index(
        drop=True
    )
    if len(radar_manifest) != 72 or radar_manifest.split.str.contains("holdout", case=False).any():
        raise RuntimeError("H1 cache requires exactly 72 TRAIN/DEV tensors")
    thermo_manifest = pd.read_csv(
        "artifacts/stage_4c/tensors/stage4c_thermodynamic_tensor_manifest.csv"
    ).set_index("anchor_id")
    apcp_sources = pd.read_csv(
        "artifacts/stage_4c/baselines/hrrr_apcp_source_manifest.csv"
    ).set_index("product_key")
    geometry_source = pd.read_csv(
        "artifacts/stage_4c/materialization/stage4_hrrr_materialized_sources.csv"
    ).iloc[0]
    _, geometry = _decode(Path(geometry_source.local_path), include_geometry=True)
    if geometry is None:
        raise RuntimeError("HRRR geometry unavailable")
    source_x, source_y = geometry
    config = yaml.safe_load(
        Path("configs/experiments/stage_4b_a_plus_radar_capacity.yaml").read_text()
    )
    metadata = json.loads(
        Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text()
    )
    radar_norm = metadata["normalization"]["radar"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_radar_model({"model": config["model"], "data": config["data"]}).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weight_cache = {}
    rows = []
    for number, row in enumerate(radar_manifest.itertuples(index=False), 1):
        payload = np.load(row.tensor_path)
        radar = payload["radar_rate_mm_hr"]
        radar_mask = payload["radar_valid_mask"].astype(np.float32)
        radar_log = (np.log1p(np.nan_to_num(radar)) - radar_norm["mean"]) / radar_norm["std"]
        radar_input = torch.from_numpy(
            np.stack([radar_log, radar_mask], 1)[None].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            logits, intensity = model(radar_input)
            expected = expected_rate_mm_hr(logits, intensity)
        logits_np = logits[0].cpu().numpy()
        intensity_np = intensity[0].cpu().numpy()
        expected_np = expected[0].cpu().numpy()
        lon, lat = payload["longitude"], payload["latitude"]
        tile = tuple(int(value) for value in payload["tile_indices"])
        if tile not in weight_cache:
            target_x, target_y = transform_coordinates(
                lon, lat, source_crs="EPSG:4326", destination_crs="EPSG:3978"
            )
            weight_cache[tile] = _weights(source_x, source_y, target_x, target_y)
        vertices, weights = weight_cache[tile]
        thermo_payload = np.load(thermo_manifest.loc[row.anchor_id].thermodynamic_tensor_path)
        records = json.loads(str(thermo_payload["source_records_json"]))
        future = [record for record in records if int(record["forecast_hour"]) in (2, 3)]
        if len(future) != 2:
            raise RuntimeError(f"{row.anchor_id}: expected f02/f03 APCP")
        hourly, source_records = [], []
        for record in future:
            source = apcp_sources.loc[record["product_key"]]
            values, units, step_range = _decode_apcp(Path(source.local_path))
            hourly.append(_apply(values, vertices, weights, lon.shape))
            source_records.append(
                {**record, "units": units, "step_range": step_range, "sha256": source.sha256}
            )
        destination = output_dir / row.split / f"{row.anchor_id}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            aplus_logits=logits_np.astype(np.float32),
            aplus_intensity_raw=intensity_np.astype(np.float32),
            aplus_expected_rate_mm_hr=expected_np.astype(np.float32),
            hrrr_apcp_hourly=np.stack(hourly).astype(np.float32),
            source_records_json=np.asarray(json.dumps(source_records)),
            aplus_checkpoint_sha256=np.asarray(checkpoint_hash),
        )
        rows.append(
            {
                "anchor_id": row.anchor_id,
                "event_id": row.event_id,
                "split": row.split,
                "event_class": row.event_class,
                "radar_goes_tensor_path": row.tensor_path,
                "stage5_cache_path": destination.as_posix(),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "aplus_logits_shape": "x".join(map(str, logits_np.shape)),
                "hrrr_apcp_shape": "x".join(map(str, np.stack(hourly).shape)),
            }
        )
        print(f"[stage5-cache] {number}/{len(radar_manifest)}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "stage5_cache_manifest.csv", index=False)
    result = {
        "rows": len(table),
        "train_rows": int(table.split.eq("train").sum()),
        "dev_rows": int(table.split.isin({"dev", "stage4_dev_repair"}).sum()),
        "aplus_checkpoint_sha256": checkpoint_hash,
        "aplus_trainable": False,
        "hrrr_intervals": [
            "1-2 hour accumulation used as 0-60 context",
            "2-3 hour accumulation used as 60-120 context",
        ],
        "protected_holdout_used": False,
        "training_started": False,
    }
    (output_dir / "stage5_cache_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
