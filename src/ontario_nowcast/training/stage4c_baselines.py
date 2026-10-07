"""Materialize and evaluate deterministic HRRR and PySTEPS+HRRR Stage 4C baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import eccodes
import numpy as np
import pandas as pd

from ..evaluation.metrics import categorical_metrics, fractions_skill_score_km
from ..models.baselines import pysteps_deterministic_extrapolation
from .stage4_materialize import (
    _download_hrrr_subset,
    _hrrr_message_ranges,
)
from .stage4c_tensors import _apply, _decode, _weights

LEADS = (30, 60, 90, 120)
THRESHOLDS = (0.1, 1.0, 2.5, 5.0)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _decode_apcp(path: Path) -> tuple[np.ndarray, str, str]:
    with path.open("rb") as handle:
        message = eccodes.codes_grib_new_from_file(handle)
        if message is None:
            raise RuntimeError(f"no GRIB message in {path}")
        try:
            short_name = str(eccodes.codes_get(message, "shortName"))
            units = str(eccodes.codes_get(message, "units"))
            step_range = str(eccodes.codes_get(message, "stepRange"))
            if short_name != "tp":
                raise RuntimeError(f"expected APCP/tp, found {short_name}")
            values = np.asarray(eccodes.codes_get_values(message), dtype=np.float32)
        finally:
            eccodes.codes_release(message)
    return values, units, step_range


def _counts_bucket() -> dict[str, Any]:
    return {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0, "fss": []}


def _update(bucket: dict[str, Any], observed: np.ndarray, forecast: np.ndarray, threshold: float) -> None:
    metric = categorical_metrics(observed, forecast, threshold)
    for name in ("hits", "misses", "false_alarms", "correct_negatives"):
        bucket[name] += int(metric[name])
    bucket["fss"].append(fractions_skill_score_km(observed, forecast, threshold, 18.0, 2.0))


def _finish(bucket: dict[str, Any]) -> dict[str, float | int | None]:
    h, m, f = bucket["hits"], bucket["misses"], bucket["false_alarms"]
    ratio = lambda n, d: float(n / d) if d else None
    return {
        "hits": h,
        "misses": m,
        "false_alarms": f,
        "correct_negatives": bucket["correct_negatives"],
        "csi": ratio(h, h + m + f),
        "pod": ratio(h, h + m),
        "far": ratio(f, h + f),
        "precision": ratio(h, h + f),
        "recall": ratio(h, h + m),
        "f1": ratio(2 * h, 2 * h + f + m),
        "fss_radius_18km": float(np.nanmean(bucket["fss"])),
        "brier_score": None,
        "brier_status": "not_applicable_deterministic_forecast",
    }


def _blend(pysteps: np.ndarray, hrrr: np.ndarray, hrrr_weight: float) -> np.ndarray:
    """Blend without allowing zero-weight NaNs to erase the selected endpoint."""
    if hrrr_weight == 0:
        return pysteps.copy()
    if hrrr_weight == 1:
        return hrrr.copy()
    return (1 - hrrr_weight) * pysteps + hrrr_weight * hrrr


def _download(product_key: str, destination: Path, manifest: Path) -> dict[str, Any]:
    if destination.exists() and destination.stat().st_size > 0:
        return {
            "product_key": product_key,
            "local_path": destination.as_posix(),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "bytes": destination.stat().st_size,
            "cached": True,
        }
    match = re.search(r"f(\d{2})\.grib2$", product_key)
    if match is None:
        raise ValueError(f"cannot infer HRRR forecast hour from {product_key}")
    forecast_hour = int(match.group(1))
    selector = f":APCP:surface:{forecast_hour - 1}-{forecast_hour} hour acc fcst:"
    ranges = _hrrr_message_ranges(product_key, {"APCP_surface": selector})
    record = _download_hrrr_subset(product_key, ranges, destination, manifest)
    return {
        "product_key": product_key,
        "local_path": destination.as_posix(),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
        "cached": bool(record["cached"]),
    }


def run(output_dir: Path) -> dict[str, Any]:
    tensor_manifest = pd.read_csv(
        "artifacts/stage_4c/tensors/stage4c_thermodynamic_tensor_manifest.csv"
    )
    dev = tensor_manifest[tensor_manifest["split"].astype(str).str.contains("dev")].copy()
    source_manifest = pd.read_csv(
        "artifacts/stage_4c/materialization/stage4_hrrr_materialized_sources.csv"
    ).set_index("product_key")
    product_by_anchor: dict[str, list[dict[str, Any]]] = {}
    product_keys: set[str] = set()
    for row in tensor_manifest.itertuples(index=False):
        payload = np.load(row.thermodynamic_tensor_path)
        records = json.loads(str(payload["source_records_json"]))
        future = [record for record in records if int(record["forecast_hour"]) in (2, 3)]
        if len(future) != 2:
            raise RuntimeError(f"{row.anchor_id}: expected HRRR forecast hours 2 and 3")
        product_by_anchor[str(row.anchor_id)] = future
        product_keys.update(str(record["product_key"]) for record in future)

    source_dir = output_dir / "apcp_sources"
    download_manifest = output_dir / "apcp_downloads.jsonl"
    source_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for product_key in sorted(product_keys):
            name = product_key.replace("/conus/", "__").replace("/", "__")
            destination = source_dir / name.replace(".grib2", ".apcp.grib2")
            futures[pool.submit(_download, product_key, destination, download_manifest)] = product_key
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index == 1 or index % 25 == 0:
                print(f"[stage4c-apcp] {index}/{len(futures)} products", flush=True)
    for row in rows:
        values, units, step_range = _decode_apcp(Path(row["local_path"]))
        row.update({
            "units": units, "step_range": step_range,
            "finite_fraction": float(np.isfinite(values).mean()),
            "minimum": float(np.nanmin(values)), "maximum": float(np.nanmax(values)),
        })
    sources = pd.DataFrame(rows).sort_values("product_key")
    sources.to_csv(output_dir / "hrrr_apcp_source_manifest.csv", index=False)
    source_paths = dict(zip(sources.product_key.astype(str), sources.local_path.astype(str)))

    first_thermo = Path(str(source_manifest.iloc[0].local_path))
    _, geometry = _decode(first_thermo, include_geometry=True)
    assert geometry is not None
    source_x, source_y = geometry
    cache: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    prediction_rows = []
    for index, row in enumerate(dev.itertuples(index=False), 1):
        radar_payload = np.load(row.radar_tensor_path)
        target_shape = radar_payload["target_rate_mm_hr"].shape[1:]
        tile = tuple(int(value) for value in radar_payload["tile_indices"])
        if tile not in cache:
            from ..preprocessing.grid import transform_coordinates

            tx, ty = transform_coordinates(
                radar_payload["longitude"], radar_payload["latitude"],
                source_crs="EPSG:4326", destination_crs="EPSG:3978",
            )
            cache[tile] = _weights(source_x, source_y, tx, ty)
        vertices, weights = cache[tile]
        hourly = []
        metadata = []
        for record in product_by_anchor[str(row.anchor_id)]:
            values, units, step_range = _decode_apcp(Path(source_paths[record["product_key"]]))
            hourly.append(_apply(values, vertices, weights, target_shape))
            metadata.append({**record, "units": units, "step_range": step_range})
        hrrr = np.stack([hourly[0], hourly[0], hourly[1], hourly[1]])
        pysteps = pysteps_deterministic_extrapolation(radar_payload["radar_rate_mm_hr"][-3:], 20)
        pysteps = np.stack([pysteps[lead // 6 - 1] for lead in LEADS])
        observed = np.stack(
            [radar_payload["target_rate_mm_hr"][lead // 6 - 1] for lead in LEADS]
        )
        valid = np.stack(
            [radar_payload["target_valid_mask"][lead // 6 - 1] > 0 for lead in LEADS]
        )
        observed = np.where(valid, observed, np.nan)
        predictions[str(row.anchor_id)] = {
            "hrrr": hrrr, "pysteps": pysteps, "observed": observed
        }
        prediction_rows.append({
            "anchor_id": row.anchor_id, "event_id": row.event_id, "split": row.split,
            "forecast_issue_time_utc": np.load(row.radar_tensor_path)["forecast_issue_time_utc"].item(),
            "source_records_json": json.dumps(metadata),
        })
        if index == 1 or index % 10 == 0:
            print(f"[stage4c-baselines] {index}/{len(dev)} DEV anchors", flush=True)

    selected: dict[int, float] = {}
    selection_rows = []
    for lead_index, lead in enumerate(LEADS):
        best = None
        for weight in BLEND_WEIGHTS:
            bucket = _counts_bucket()
            for prediction in predictions.values():
                blend = _blend(
                    prediction["pysteps"][lead_index], prediction["hrrr"][lead_index], weight
                )
                _update(bucket, prediction["observed"][lead_index], blend, 0.1)
            metric = _finish(bucket)
            selection_rows.append({"lead_minutes": lead, "hrrr_weight": weight, **metric})
            score = -1.0 if metric["f1"] is None else float(metric["f1"])
            candidate = (score, -abs(weight - 0.5), -weight, weight)
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        selected[lead] = float(best[-1])

    buckets: dict[tuple[str, int, float], dict[str, Any]] = defaultdict(_counts_bucket)
    event_buckets: dict[tuple[str, str, int, float], dict[str, Any]] = defaultdict(_counts_bucket)
    event_lookup = dict(zip(dev.anchor_id.astype(str), dev.event_id.astype(str)))
    for anchor_id, prediction in predictions.items():
        for lead_index, lead in enumerate(LEADS):
            forecasts = {
                "raw_hrrr_apcp": prediction["hrrr"][lead_index],
                "pysteps": prediction["pysteps"][lead_index],
                "pysteps_hrrr_dev_blend": _blend(
                    prediction["pysteps"][lead_index],
                    prediction["hrrr"][lead_index],
                    selected[lead],
                ),
            }
            for name, forecast in forecasts.items():
                for threshold in THRESHOLDS:
                    _update(buckets[(name, lead, threshold)], prediction["observed"][lead_index], forecast, threshold)
                    _update(event_buckets[(event_lookup[anchor_id], name, lead, threshold)], prediction["observed"][lead_index], forecast, threshold)
    metrics = pd.DataFrame([
        {"model": name, "lead_minutes": lead, "threshold_mm_hr": threshold, **_finish(bucket)}
        for (name, lead, threshold), bucket in sorted(buckets.items())
    ])
    event_metrics = pd.DataFrame([
        {"event_id": event, "model": name, "lead_minutes": lead, "threshold_mm_hr": threshold, **_finish(bucket)}
        for (event, name, lead, threshold), bucket in sorted(event_buckets.items())
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "dev_prediction_provenance.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / "dev_blend_weight_search.csv", index=False)
    metrics.to_csv(output_dir / "dev_baseline_metrics.csv", index=False)
    event_metrics.to_csv(output_dir / "dev_baseline_event_metrics.csv", index=False)
    summary = {
        "scope": "DEV only",
        "dev_anchors": len(dev),
        "hrrr_products": len(sources),
        "hrrr_interpretation": "APCP hourly accumulation; mm per hour interval equals interval-average mm/h",
        "lead_mapping": {"30": "f02 1-2h APCP", "60": "f02 1-2h APCP", "90": "f03 2-3h APCP", "120": "f03 2-3h APCP"},
        "selected_hrrr_blend_weight_by_lead": selected,
        "blend_selection_metric": "pooled DEV F1 at 0.1 mm/h on five fixed weights",
        "probabilistic_metrics": "not applicable; no probability manufactured",
        "final_holdout_scored": False,
        "stage_c_training_started": False,
    }
    (output_dir / "stage4c_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4c/baselines"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
