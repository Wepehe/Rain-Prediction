"""Exact post-freeze Stage 4B holdout materialization and scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ..data.manifest import download_immutable, sha256_file
from ..data.multimodal import _goes_scan_time, nearest_goes_east_channel13
from ..data.s3 import object_url
from ..evaluation.metrics import brier_score
from ..evaluation.onset import first_crossing_minutes, onset_metrics
from ..models.baselines import pysteps_deterministic_extrapolation
from ..models.radar_convlstm import build_radar_model
from ..models.radar_goes import build_stage4b_model
from ..preprocessing.fuse import _interpolate_goes
from .stage4_materialize import _cooling_tendency, _load_event_radar
from .stage4b_lifecycle import _goes_inputs, _radar_inputs
from .stage4b_train import _load_npz

HOLDOUT = Path("artifacts/stage_4/gate/stage4_final_initiation_holdout_manifest.csv")
FROZEN = Path("artifacts/stage_4b/stage_4b_frozen_procedure_manifest.json")
EXPECTED_HASH = "0d347e2bff4ab174af248b2345e709f0c0b12505a0542dcca573e10d2af08783"
RADAR_LAGS = (54, 48, 42, 36, 30, 24, 18, 12, 6, 0)
GOES_LAGS = (60, 30, 20, 10, 0)
COOLING = (10, 20, 30, 60)
RAIN = 0.1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _source(requested: pd.Timestamp, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = requested.isoformat()
    if key in cache:
        return cache[key]
    prefix = f"*C13_*_s{requested:%Y%j%H}*.nc"
    local_candidates = []
    for bucket_dir in (Path("data/raw/satellite/noaa-goes19"), Path("data/raw/satellite/noaa-goes16")):
        for path in bucket_dir.glob(prefix):
            actual_local = pd.Timestamp(_goes_scan_time(path.name))
            if actual_local <= requested:
                local_candidates.append((actual_local, path))
    if local_candidates:
        actual, destination = max(local_candidates, key=lambda item: item[0])
        if requested-actual <= pd.Timedelta(minutes=6):
            cache[key] = {"requested": key, "actual": actual.isoformat(), "path": destination.as_posix(), "offset_minutes": (actual-requested).total_seconds()/60}
            return cache[key]
    bucket, obj = nearest_goes_east_channel13(requested.to_pydatetime())
    actual = pd.Timestamp(_goes_scan_time(obj.key))
    # Causality: never use a scan beginning after the requested availability time.
    if actual > requested:
        previous = requested - pd.Timedelta(minutes=5)
        bucket, obj = nearest_goes_east_channel13(previous.to_pydatetime())
        actual = pd.Timestamp(_goes_scan_time(obj.key))
    if actual > requested:
        raise RuntimeError(f"no causal GOES scan for {requested.isoformat()}")
    destination = Path("data/raw/satellite") / bucket / Path(obj.key).name
    if not destination.exists():
        download_immutable(
            object_url(bucket, obj.key),
            destination,
            Path("data/metadata/downloads.jsonl"),
            metadata={"source": f"NOAA {bucket}", "product": "ABI-L2-CMIPC C13", "stage": "stage_4b_final_holdout"},
        )
    cache[key] = {"requested": key, "actual": actual.isoformat(), "path": destination.as_posix(), "offset_minutes": (actual-requested).total_seconds()/60}
    return cache[key]


def materialize(output: Path) -> dict[str, Any]:
    if _sha(HOLDOUT) != EXPECTED_HASH:
        raise RuntimeError("frozen holdout manifest hash changed")
    frozen = json.loads(FROZEN.read_text())
    if frozen["holdout_manifest_sha256"] != EXPECTED_HASH:
        raise RuntimeError("frozen procedure references a different holdout")
    rows = pd.read_csv(HOLDOUT)
    clean = rows[rows.clean_pre_radar_initiation.astype(str).str.lower().eq("true")].copy()
    sources: dict[str, dict[str, Any]] = {}
    required_times = set()
    for value in clean.representative_issue_time_utc:
        issue = pd.Timestamp(value).tz_convert("UTC")
        for lag in GOES_LAGS:
            current = issue-pd.Timedelta(minutes=10+lag)
            required_times.add(current)
            required_times.update(current-pd.Timedelta(minutes=m) for m in COOLING)
    print(f"[holdout-materialize] prefetching {len(required_times)} unique causal GOES times", flush=True)
    # NOAA public S3 begins throttling this workload above a few concurrent listings.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_source, value, sources): value for value in sorted(required_times)}
        for number, future in enumerate(as_completed(futures), 1):
            future.result()
            if number == 1 or number % 25 == 0:
                print(f"[holdout-materialize] GOES {number}/{len(futures)}", flush=True)
    event_cache: dict[str, tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]] = {}
    out_rows, failures = [], []
    tensor_dir = output / "tensors"
    for number, row in enumerate(clean.itertuples(index=False), 1):
        identity = f"{row.source_event_id}__{row.object_id}__{pd.Timestamp(row.representative_issue_time_utc):%Y%m%dT%H%M%SZ}"
        try:
            issue = pd.Timestamp(row.representative_issue_time_utc).tz_convert("UTC")
            if row.source_event_id not in event_cache:
                event_cache[row.source_event_id] = _load_event_radar(row.source_event_id, Path("data"))
            rates, times, lat, lon = event_cache[row.source_event_id]
            y0, y1, x0, x1 = (
                round(float(x))
                for x in (row.tile_y_min, row.tile_y_max, row.tile_x_min, row.tile_x_max)
            )
            if (y1-y0+1, x1-x0+1) != (128, 128):
                raise ValueError("frozen tile is not 128x128")
            destination = tensor_dir / f"{identity}.npz"
            if destination.exists():
                existing = np.load(destination)
                radar, goes, target = existing["radar_rate_mm_hr"], existing["goes"], existing["target_rate_mm_hr"]
                out_rows.append({"identity": identity, "event_id": row.source_event_id, "object_id": row.object_id, "issue_time_utc": issue.isoformat(), "tensor_path": destination.as_posix(), "sha256": sha256_file(destination), "tile_y_min": y0, "tile_y_max": y1, "tile_x_min": x0, "tile_x_max": x1, "radar_shape": str(radar.shape), "goes_shape": str(goes.shape), "target_shape": str(target.shape), "radar_valid_fraction": float(np.isfinite(radar).mean()), "goes_valid_fraction": float(np.isfinite(goes).mean()), "target_valid_fraction": float(np.isfinite(target).mean()), "future_radar_in_inputs": False, "latest_goes_before_issue_minutes": float((issue-max(pd.Timestamp(str(t)) for t in existing["goes_actual_source_times_utc"])).total_seconds()/60)})
                print(f"[holdout-materialize] {number}/{len(clean)} cached {identity}", flush=True)
                continue
            wanted_radar = [issue-pd.Timedelta(minutes=m) for m in RADAR_LAGS]
            wanted_target = [issue+pd.Timedelta(minutes=m) for m in range(6, 121, 6)]
            lookup = {t: i for i, t in enumerate(times)}
            missing = [t for t in wanted_radar+wanted_target if t not in lookup]
            if missing:
                raise KeyError(f"missing exact radar times: {[x.isoformat() for x in missing]}")
            radar = np.stack([rates[lookup[t], y0:y1+1, x0:x1+1] for t in wanted_radar]).astype(np.float32)
            target = np.stack([rates[lookup[t], y0:y1+1, x0:x1+1] for t in wanted_target]).astype(np.float32)
            tile_lon, tile_lat = np.meshgrid(lon[x0:x1+1], lat[y0:y1+1])
            raw, actual_times = [], []
            cooling = {m: [] for m in COOLING}
            for lag in GOES_LAGS:
                requested = issue-pd.Timedelta(minutes=10+lag)
                current_source = _source(requested, sources)
                current = _interpolate_goes(Path(current_source["path"]), tile_lon, tile_lat)
                raw.append(current)
                actual_times.append(current_source["actual"])
                for minutes in COOLING:
                    base_source = _source(requested-pd.Timedelta(minutes=minutes), sources)
                    baseline = _interpolate_goes(Path(base_source["path"]), tile_lon, tile_lat)
                    cooling[minutes].append(_cooling_tendency(current, baseline))
            goes = np.stack([np.stack(raw), *[np.stack(cooling[m]) for m in COOLING]]).astype(np.float32)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, radar_rate_mm_hr=radar, radar_valid_mask=np.isfinite(radar).astype(np.float32), target_rate_mm_hr=target, target_valid_mask=np.isfinite(target).astype(np.float32), goes=np.nan_to_num(goes, nan=0).astype(np.float32), goes_valid_mask=np.isfinite(goes).astype(np.float32), forecast_issue_time_utc=np.asarray(issue.isoformat()), object_id=np.asarray(row.object_id), source_event_id=np.asarray(row.source_event_id), tile_indices=np.asarray([y0,y1,x0,x1]), goes_actual_source_times_utc=np.asarray(actual_times))
            out_rows.append({"identity": identity, "event_id": row.source_event_id, "object_id": row.object_id, "issue_time_utc": issue.isoformat(), "tensor_path": destination.as_posix(), "sha256": sha256_file(destination), "tile_y_min": y0, "tile_y_max": y1, "tile_x_min": x0, "tile_x_max": x1, "radar_shape": str(radar.shape), "goes_shape": str(goes.shape), "target_shape": str(target.shape), "radar_valid_fraction": float(np.isfinite(radar).mean()), "goes_valid_fraction": float(np.isfinite(goes).mean()), "target_valid_fraction": float(np.isfinite(target).mean()), "future_radar_in_inputs": False, "latest_goes_before_issue_minutes": float((issue-max(pd.Timestamp(t) for t in actual_times)).total_seconds()/60)})
            print(f"[holdout-materialize] {number}/{len(clean)} {identity}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"identity": identity, "event_id": row.source_event_id, "object_id": row.object_id, "issue_time_utc": row.representative_issue_time_utc, "error": str(exc)})
            print(f"[holdout-materialize] FAILED {identity}: {exc}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(out_rows)
    manifest.to_csv(output/"exact_holdout_tensor_manifest.csv", index=False)
    pd.DataFrame(failures).to_csv(output/"exact_holdout_tensor_failures.csv", index=False)
    pd.DataFrame(sources.values()).to_csv(output/"exact_holdout_goes_sources.csv", index=False)
    integrity = {"created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(), "frozen_manifest_sha256": _sha(HOLDOUT), "expected_clean_rows": 68, "materialized_rows": len(manifest), "failures": len(failures), "exact_issue_times": bool(len(manifest)==68 and set(manifest.issue_time_utc)==set(pd.to_datetime(clean.representative_issue_time_utc, utc=True).map(lambda x:x.isoformat()))), "dimensions_valid": bool(len(manifest)==68 and (manifest.radar_shape=="(10, 128, 128)").all() and (manifest.goes_shape=="(5, 5, 128, 128)").all() and (manifest.target_shape=="(20, 128, 128)").all()), "causal_inputs": bool(len(manifest)==68 and (~manifest.future_radar_in_inputs).all() and (manifest.latest_goes_before_issue_minutes>=10).all()), "complete_and_scoreable": bool(len(manifest)==68 and not failures)}
    (output/"materialization_integrity_report.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    return integrity


def _load_models(device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    configs = {
        "stage31": _yaml(Path("configs/experiments/stage_3_1_radar_control.yaml")),
        "A_plus": _yaml(Path("configs/experiments/stage_4b_a_plus_radar_capacity.yaml")),
        "B1": _yaml(Path("configs/experiments/stage_4b_b1_raw_c13.yaml")),
        "B2": _yaml(Path("configs/experiments/stage_4b_b2_c13_cooling.yaml")),
    }
    models = {
        "stage31": build_radar_model(configs["stage31"]).to(device),
        "A_plus": build_radar_model({"model": configs["A_plus"]["model"], "data": configs["A_plus"]["data"]}).to(device),
        "B1": build_stage4b_model(configs["B1"]).to(device),
        "B2": build_stage4b_model(configs["B2"]).to(device),
    }
    checkpoints = {
        "stage31": "artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/model_best.pt",
        "A_plus": "artifacts/stage_4b/a_plus_radar/model_best.pt",
        "B1": "artifacts/stage_4b/b1_raw_c13/model_best.pt",
        "B2": "artifacts/stage_4b/b2_c13_cooling/model_best.pt",
    }
    for name, model in models.items():
        model.load_state_dict(torch.load(checkpoints[name], map_location=device))
        model.eval()
    metadata = {
        name: json.loads(Path(f"artifacts/stage_4b/{folder}/training_metadata.json").read_text())
        for name, folder in (("A_plus", "a_plus_radar"), ("B1", "b1_raw_c13"), ("B2", "b2_c13_cooling"))
    }
    return models, {"configs": configs, "metadata": metadata}


def _row_metrics(observed: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    obs_minutes = first_crossing_minutes(observed, RAIN, 6)
    pred_minutes = first_crossing_minutes(probability, threshold, 6)
    raw = onset_metrics(obs_minutes.ravel(), pred_minutes.ravel())
    hit, false, miss = int(raw["paired_events"]), int(raw["false_events"]), int(raw["missed_events"])
    precision, recall = hit/max(hit+false, 1), hit/max(hit+miss, 1)
    result = {"precision": precision, "recall": recall, "f1": 2*precision*recall/max(precision+recall, 1e-12), "false_initiation_fraction": false/max(hit+false, 1), "onset_mae_minutes": float(raw["mae_minutes"]), "median_onset_bias_minutes": float(raw["median_error_minutes"]), "brier_score": brier_score(np.isfinite(obs_minutes).astype(float), np.nanmax(probability, axis=0))}
    for window in (30, 60, 90, 120):
        eligible = np.isfinite(obs_minutes) & (obs_minutes <= window)
        result[f"detection_within_{window}min"] = float(np.mean(np.isfinite(pred_minutes[eligible]) & (pred_minutes[eligible] <= window))) if eligible.any() else float("nan")
    return result


def score(output: Path) -> dict[str, Any]:
    integrity = json.loads((output/"materialization_integrity_report.json").read_text())
    if not integrity["complete_and_scoreable"]:
        raise RuntimeError("exact holdout integrity gate did not pass; scoring refused")
    manifest = pd.read_csv(output/"exact_holdout_tensor_manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, details = _load_models(device)
    normalization = details["metadata"]["B2"]["normalization"]
    thresholds = {"stage31": .05, "A_plus": .35, "B1": .30, "B2": .20, "PySTEPS": .5}
    rows, strict_rows = [], []
    for number, record in enumerate(manifest.itertuples(index=False), 1):
        payload = _load_npz(Path(record.tensor_path))
        radar_np = _radar_inputs(payload, normalization)
        radar = torch.from_numpy(radar_np[None]).to(device)
        outputs: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for name in ("stage31", "A_plus"):
                logits, _ = models[name](radar)
                outputs[name] = torch.sigmoid(logits)[0, :, 0].cpu().numpy()
            for name in ("B1", "B2"):
                channel_indices = details["configs"][name]["data"]["goes_channel_indices"]
                goes_np = _goes_inputs(payload, details["metadata"][name]["normalization"], channel_indices=channel_indices)
                logits, _ = models[name](radar, torch.from_numpy(goes_np[None]).to(device))
                outputs[name] = torch.sigmoid(logits)[0, :, 0].cpu().numpy()
        pysteps_rate = pysteps_deterministic_extrapolation(payload["radar"][-3:], 20)
        outputs["PySTEPS"] = (pysteps_rate >= RAIN).astype(np.float32)
        latest = payload["radar"][-1]
        dry = np.isfinite(latest) & (latest < RAIN)
        observed = np.where(payload["target_mask"] > 0, payload["target"], np.nan)[:, dry]
        for name, probability in outputs.items():
            rows.append({"identity": record.identity, "event_id": record.event_id, "object_id": record.object_id, "model": name, "threshold": thresholds[name], **_row_metrics(observed, probability[:, dry], thresholds[name])})
        eventual = np.any(observed >= RAIN, axis=0)
        py_event_rain = float(np.mean(np.any(pysteps_rate[:, dry][:, eventual] >= RAIN, axis=0))) if eventual.any() else 0.0
        strict_rows.append({"identity": record.identity, "event_id": record.event_id, "object_id": record.object_id, "history_all_dry_at_eventual_pixels": bool(np.all(payload["radar"][:, dry][:, eventual] < RAIN)) if eventual.any() else False, "advective_entry_like": False, "pysteps_eventual_pixel_rain_fraction": py_event_rain, "radar_coverage_valid": bool(np.all(payload["radar_mask"][:, dry][:, eventual] > 0)) if eventual.any() else False, "strict_subset": bool(eventual.any() and np.all(payload["radar"][:, dry][:, eventual] < RAIN) and py_event_rain <= .05 and np.all(payload["radar_mask"][:, dry][:, eventual] > 0))})
        print(f"[holdout-score] {number}/{len(manifest)} {record.identity}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/"clean_initiation_metrics_by_row.csv", index=False)
    strict = pd.DataFrame(strict_rows)
    strict.to_csv(output/"strict_subset_membership.csv", index=False)
    macro = table.groupby(["model", "event_id"], as_index=False).mean(numeric_only=True)
    macro.to_csv(output/"clean_initiation_metrics_by_event.csv", index=False)
    pooled = table.groupby("model", as_index=False).mean(numeric_only=True)
    pooled.to_csv(output/"clean_initiation_metrics_summary.csv", index=False)
    strict_metrics = table.merge(strict[strict.strict_subset][["identity"]], on="identity").groupby("model", as_index=False).mean(numeric_only=True)
    strict_metrics.to_csv(output/"strict_subset_metrics_summary.csv", index=False)
    summary = {"scored_at_utc": pd.Timestamp.now(tz=UTC).isoformat(), "rows": len(manifest), "events": int(manifest.event_id.nunique()), "strict_rows": int(strict.strict_subset.sum()), "strict_events": int(strict[strict.strict_subset].event_id.nunique()), "dev_selection_reopened": False, "calibration": "identity", "metrics_summary": (output/"clean_initiation_metrics_summary.csv").as_posix()}
    (output/"final_holdout_scoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def score_hard_negatives(output: Path) -> dict[str, Any]:
    tensor_manifest = pd.read_csv("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv")
    negatives = tensor_manifest[(tensor_manifest.split == "stage4_initiation_holdout") & (tensor_manifest.event_class == "hard_negative_candidate")].copy()
    if len(negatives) != 16 or negatives.event_id.nunique() != 4:
        raise RuntimeError("expected 16 protected hard-negative tensors across four events")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, details = _load_models(device)
    normalization = details["metadata"]["B2"]["normalization"]
    thresholds = {"stage31": .05, "A_plus": .35, "B2": .20, "PySTEPS": .5}
    rows = []
    for number, record in enumerate(negatives.itertuples(index=False), 1):
        payload = _load_npz(Path(record.tensor_path))
        radar = torch.from_numpy(_radar_inputs(payload, normalization)[None]).to(device)
        with torch.no_grad():
            outputs = {}
            for name in ("stage31", "A_plus"):
                logits, _ = models[name](radar)
                outputs[name] = torch.sigmoid(logits)[0, :, 0].cpu().numpy()
            goes_np = _goes_inputs(payload, details["metadata"]["B2"]["normalization"], channel_indices=[0,1,2,3,4])
            logits, _ = models["B2"](radar, torch.from_numpy(goes_np[None]).to(device))
            outputs["B2"] = torch.sigmoid(logits)[0, :, 0].cpu().numpy()
        py_rate = pysteps_deterministic_extrapolation(payload["radar"][-3:], 20)
        outputs["PySTEPS"] = (py_rate >= RAIN).astype(np.float32)
        observed = np.where(payload["target_mask"] > 0, payload["target"], np.nan)
        observed_event = np.any(observed >= RAIN, axis=0)
        for name, probability in outputs.items():
            event_probability = np.nanmax(probability, axis=0)
            predicted_event = event_probability >= thresholds[name]
            false_pixels = (~observed_event) & predicted_event
            predicted_pixels = predicted_event
            rows.append({"sample_id": record.anchor_id, "event_id": record.event_id, "model": name, "threshold": thresholds[name], "mean_rain_probability": float(np.nanmean(probability)), "max_rain_probability": float(np.nanmax(probability)), "wet_area_fraction_above_operational_threshold": float(np.nanmean(probability >= thresholds[name])), "false_initiation_fraction": float(false_pixels.sum()/max(predicted_pixels.sum(), 1)), "brier_score": brier_score((observed >= RAIN).astype(float), probability)})
        print(f"[holdout-hard-negative] {number}/{len(negatives)} {record.anchor_id}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output/"hard_negative_metrics_by_sample.csv", index=False)
    by_event = table.groupby(["model", "event_id"], as_index=False).mean(numeric_only=True)
    by_event.to_csv(output/"hard_negative_metrics_by_event.csv", index=False)
    summary_table = by_event.groupby("model", as_index=False).mean(numeric_only=True)
    summary_table.to_csv(output/"hard_negative_metrics_summary.csv", index=False)
    summary = {"samples": len(negatives), "events": int(negatives.event_id.nunique()), "models": sorted(table.model.unique()), "summary": (output/"hard_negative_metrics_summary.csv").as_posix()}
    (output/"hard_negative_scoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("materialize", "score", "hard-negatives"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4b/final_holdout"))
    args = parser.parse_args()
    action = {"materialize": materialize, "score": score, "hard-negatives": score_hard_negatives}[args.mode]
    print(json.dumps(action(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
