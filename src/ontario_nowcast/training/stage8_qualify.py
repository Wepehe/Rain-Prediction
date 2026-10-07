"""Qualify and freeze Stage 8 event support without forecast-model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data.manifest import sha256_file
from ..events.tracking import build_independent_initiation_catalog
from ..sample_evaluation import _downsample_max
from .stage4_gate import _tile_bounds, _tile_wgs84
from .stage4c_baselines import _decode_apcp, _download


def _radar_path(event_id: str) -> Path:
    candidates = [
        Path("data/stage8_qualification") / event_id / "processed/events" / f"{event_id}.npz",
        Path("data/s8/e01/processed/events") / f"{event_id}.npz",
        Path("data/s8/e02/processed/events") / f"{event_id}.npz",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def _sha_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _system_group(event_id: str) -> tuple[str, str, str]:
    if event_id in {"stage8_add_dev_sep07_2021", "stage8_add_dev_sep08_2021"}:
        return (
            "dev_2021_09_07_08_shared_system",
            "September 7 and 8 are conservatively treated as one continuous adjacent storm system.",
            "Northern Tornadoes Project 2021 event index; adjacent dates and requested cap",
        )
    return (
        event_id.replace("stage7_", "").replace("stage8_add_", ""),
        "Temporally separated predeclared event period; no overlap with another candidate system.",
        "predeclared candidate manifest and non-overlapping event window",
    )


def _object_rows(event: dict, split: str, config: dict, support_dir: Path):
    event_id = event["id"]
    path = _radar_path(event_id)
    payload = np.load(path)
    native = payload["rate_mm_hr"]
    times = pd.to_datetime(payload["times"], utc=True)
    source_files = payload["source_files"].astype(str)
    rates = _downsample_max(native, 2)
    lat = payload["latitude"][: rates.shape[1] * 2 : 2]
    lon = payload["longitude"][: rates.shape[2] * 2 : 2]
    missing_frames = int(np.count_nonzero(source_files == ""))
    missing_fraction = missing_frames / len(times)
    catalog = build_independent_initiation_catalog(
        rates,
        times,
        lat,
        lon,
        interval_minutes=6,
        resolution_km=2.0,
        history_minutes=60,
        horizon_minutes=120,
        rain_threshold=0.1,
        strong_threshold=5.0,
        max_missing_fraction=0.1,
        minimum_pixels=12,
    )
    group_id, rationale, evidence = _system_group(event_id)
    rows = []
    boundary_failures = validity_failures = advective = 0
    support_dir.mkdir(parents=True, exist_ok=True)
    for index, item in catalog.reset_index(drop=True).iterrows():
        issue = pd.Timestamp(item["representative_anchor_time"])
        anchor = int(times.get_indexer([issue])[0])
        if anchor < 9 or anchor + 20 >= len(times):
            validity_failures += 1
            continue
        yc = (float(item["y_min"]) + float(item["y_max"])) / 2
        xc = (float(item["x_min"]) + float(item["x_max"])) / 2
        y0, y1 = _tile_bounds(yc, 128, rates.shape[1])
        x0, x1 = _tile_bounds(xc, 128, rates.shape[2])
        history = rates[anchor - 9 : anchor + 1, y0 : y1 + 1, x0 : x1 + 1]
        target = rates[anchor + 1 : anchor + 21, y0 : y1 + 1, x0 : x1 + 1]
        finite = float(np.isfinite(np.concatenate([history, target])).mean())
        apparent = bool(item["apparent_initiation"])
        entry = bool(item["advective_entry_like"])
        boundary = bool(item["touches_domain_boundary"])
        clean_geometry = apparent and not entry and not boundary and float(item["future_max_mm_hr"]) >= 1
        advective += int(entry)
        boundary_failures += int(boundary)
        valid = finite >= 0.90 and missing_fraction <= 0.10
        validity_failures += int(not valid)
        clean = clean_geometry and valid
        object_id = f"{event_id}__{item['event_id']}"
        mask_path = support_dir / f"{object_id}.masks.npz"
        if clean:
            np.savez_compressed(
                mask_path,
                radar_history_valid_mask=np.isfinite(history),
                target_valid_mask=np.isfinite(target),
                radar_history_times=np.asarray(times[anchor - 9 : anchor + 1].astype(str)),
                target_times=np.asarray(times[anchor + 1 : anchor + 21].astype(str)),
                source_radar_sha256=np.asarray(sha256_file(path)),
            )
        rows.append(
            {
                "split": split,
                "event_id": event_id,
                "independent_system_group": group_id,
                "object_id": object_id,
                "issue_time_utc": issue.isoformat(),
                "tile_y_min": y0,
                "tile_y_max": y1,
                "tile_x_min": x0,
                "tile_x_max": x1,
                "tile_bbox_wgs84": _tile_wgs84(lat, lon, y_min=y0, y_max=y1, x_min=x0, x_max=x1),
                "radar_history_times": "|".join(times[anchor - 9 : anchor + 1].astype(str)),
                "target_times": "|".join(times[anchor + 1 : anchor + 21].astype(str)),
                "radar_valid_fraction": finite,
                "radar_source_path": path.as_posix(),
                "radar_source_sha256": sha256_file(path),
                "validity_mask_path": mask_path.as_posix() if clean else "",
                "validity_mask_sha256": sha256_file(mask_path) if clean else "",
                "future_max_mm_hr": float(item["future_max_mm_hr"]),
                "apparent_initiation": apparent,
                "advective_entry_like": entry,
                "touches_domain_boundary": boundary,
                "clean_initiation": clean,
                "pysteps_eligible": bool(clean and np.isfinite(history[-3:]).mean() >= 0.90),
                "a_plus_eligible": bool(clean and history.shape == (10, 128, 128)),
            }
        )
    clean_rows = [row for row in rows if row["clean_initiation"]]
    summary = {
        "split": split,
        "event_id": event_id,
        "radar_path": path.as_posix(),
        "radar_sha256": sha256_file(path),
        "expected_frames": len(times),
        "available_frames": len(times) - missing_frames,
        "missing_frames": missing_frames,
        "missing_fraction": missing_fraction,
        "timestamp_start": times[0].isoformat(),
        "timestamp_end": times[-1].isoformat(),
        "timestamp_exact": bool(times.is_monotonic_increasing and len(times.unique()) == len(times)),
        "units": "mm h-1",
        "spatial_shape_native": f"{native.shape[1]}x{native.shape[2]}",
        "finite_fraction": float(np.isfinite(native).mean()),
        "initiation_like_objects": len(catalog),
        "clean_initiation_objects": len(clean_rows),
        "unique_clean_issue_times": len({row["issue_time_utc"] for row in clean_rows}),
        "advective_entries": advective,
        "boundary_failures": boundary_failures,
        "radar_validity_failures": validity_failures,
        "radar_quality_pass": missing_fraction <= 0.10,
    }
    group = {
        "split": split,
        "candidate_period": event_id,
        "group_id": group_id,
        "rationale": rationale,
        "source_evidence": evidence,
    }
    return rows, summary, group, rates, times, lat, lon


def _hard_negatives(event_id: str, split: str, group_id: str, rates, times, lat, lon):
    rows = []
    centres = [(64, 64), (64, rates.shape[2] - 65), (rates.shape[1] - 65, 64), (rates.shape[1] - 65, rates.shape[2] - 65)]
    for anchor in range(9, len(times) - 20, 20):
        for yc, xc in centres:
            y0, y1 = _tile_bounds(yc, 128, rates.shape[1]); x0, x1 = _tile_bounds(xc, 128, rates.shape[2])
            window = rates[anchor - 9 : anchor + 21, y0 : y1 + 1, x0 : x1 + 1]
            valid = np.isfinite(window)
            finite = float(valid.mean())
            wet_by_frame = np.mean(valid & (window >= 0.1), axis=(1, 2))
            heavy = float(np.mean(valid & (window >= 5.0)))
            if finite >= 0.90 and float(wet_by_frame.mean()) <= 0.001 and float(wet_by_frame.max()) <= 0.005 and heavy == 0:
                rows.append({
                    "split": split, "event_id": event_id, "independent_system_group": group_id,
                    "hard_negative_id": f"{event_id}__dry_{len(rows)+1:02d}",
                    "issue_time_utc": times[anchor].isoformat(), "tile_y_min": y0, "tile_y_max": y1,
                    "tile_x_min": x0, "tile_x_max": x1,
                    "tile_bbox_wgs84": _tile_wgs84(lat, lon, y_min=y0, y_max=y1, x_min=x0, x_max=x1),
                    "mean_wet_fraction": float(wet_by_frame.mean()), "maximum_wet_fraction": float(wet_by_frame.max()),
                    "heavy_rain_fraction": heavy, "radar_valid_fraction": finite, "temporal_frames": len(window),
                })
                break
        if len(rows) >= 2:
            break
    return rows


def _hrrr_products(issue_times: list[str], output_dir: Path):
    requirements = []
    products = set()
    for issue_text in sorted(set(issue_times)):
        issue = pd.Timestamp(issue_text)
        cycle = (issue - pd.Timedelta(minutes=60)).floor("h")
        available = cycle + pd.Timedelta(minutes=60)
        for hour in (2, 3):
            product = f"hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfsfcf{hour:02d}.grib2"
            requirements.append({
                "issue_time_utc": issue.isoformat(), "hrrr_cycle_issue_time_utc": cycle.isoformat(),
                "simulated_availability_time_utc": available.isoformat(),
                "forecast_valid_time_utc": (cycle + pd.Timedelta(hours=hour)).isoformat(),
                "forecast_hour": hour, "product_identity": product,
                "causal": bool(available <= issue),
            })
            products.add(product)
    source_dir = output_dir / "hrrr_apcp_sources"; source_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for product in sorted(products):
            destination = source_dir / product.replace("/conus/", "__").replace("/", "__").replace(".grib2", ".apcp.grib2")
            futures[pool.submit(_download, product, destination, output_dir / "hrrr_downloads.jsonl")] = product
        for future in as_completed(futures):
            rows.append(future.result())
    source = pd.DataFrame(rows)
    for row in source.to_dict("records"):
        values, units, step_range = _decode_apcp(Path(row["local_path"]))
        source.loc[source.product_key == row["product_key"], "units"] = units
        source.loc[source.product_key == row["product_key"], "step_range"] = step_range
        source.loc[source.product_key == row["product_key"], "finite_fraction"] = float(np.isfinite(values).mean())
    lookup = source.set_index("product_key")
    for row in requirements:
        record = lookup.loc[row["product_identity"]]
        row.update({"local_path": record.local_path, "checksum_sha256": record.sha256, "source_finite_fraction": record.finite_fraction})
    return pd.DataFrame(requirements), source.sort_values("product_key")


def run(config_path: Path, output_dir: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_objects, summaries, groups, negatives = [], [], [], []
    support_dir = output_dir / "object_support"
    for key, split in (("development_events", "development"), ("final_events", "final")):
        for event in config[key]:
            rows, summary, group, rates, times, lat, lon = _object_rows(event, split, config, support_dir)
            all_objects.extend(rows); summaries.append(summary); groups.append(group)
            negatives.extend(_hard_negatives(event["id"], split, group["group_id"], rates, times, lat, lon))
    objects = pd.DataFrame(all_objects)
    summary = pd.DataFrame(summaries)
    group_table = pd.DataFrame(groups)
    minimum_issues = int(config["positive_event_rule"]["minimum_distinct_clean_issue_times"])
    summary["positive_period_qualified"] = summary.radar_quality_pass & (summary.unique_clean_issue_times >= minimum_issues)
    qualified = set(summary.loc[summary.positive_period_qualified, "event_id"])
    group_table["positive_period_qualified"] = group_table.candidate_period.isin(qualified)
    group_table["independent_event_credit"] = 0
    for (_, group), frame in group_table.groupby(["split", "group_id"]):
        eligible = frame[frame.positive_period_qualified]
        if len(eligible):
            group_table.loc[eligible.index[0], "independent_event_credit"] = 1
    clean = objects[objects.clean_initiation & objects.event_id.isin(qualified)].copy()
    hard = pd.DataFrame(negatives)
    hard = hard[hard.event_id.isin(qualified)].copy() if len(hard) else hard
    issue_times = clean.issue_time_utc.astype(str).tolist() + (hard.issue_time_utc.astype(str).tolist() if len(hard) else [])
    hrrr, hrrr_sources = _hrrr_products(issue_times, output_dir)
    causal = hrrr.groupby("issue_time_utc").agg(hrrr_records=("product_identity", "size"), hrrr_all_causal=("causal", "all"), hrrr_min_finite=("source_finite_fraction", "min")).reset_index()
    clean = clean.merge(causal, on="issue_time_utc", how="left")
    clean["hrrr_qualified"] = clean.hrrr_all_causal & (clean.hrrr_records == 2) & (clean.hrrr_min_finite >= 0.99)
    clean = clean[clean.hrrr_qualified].copy()
    summary.to_csv(output_dir / "stage8_candidate_qualification.csv", index=False)
    objects.to_csv(output_dir / "stage8_all_mined_objects.csv", index=False)
    group_table.to_csv(output_dir / "stage8_independent_system_groups.csv", index=False)
    hard.to_csv(output_dir / "stage8_hard_negative_manifest.csv", index=False)
    hrrr.to_csv(output_dir / "stage8_hrrr_causal_records.csv", index=False)
    hrrr_sources.to_csv(output_dir / "stage8_hrrr_source_manifest.csv", index=False)
    rejection = summary.loc[~summary.positive_period_qualified].copy()
    rejection["rejection_reason"] = np.where(~rejection.radar_quality_pass, "radar_missing_fraction_exceeds_0.10", "fewer_than_3_clean_issue_times")
    rejection.to_csv(output_dir / "stage8_rejected_candidates.csv", index=False)
    counts = group_table.groupby("split").independent_event_credit.sum().to_dict()
    final_groups = int(counts.get("final", 0)); dev_groups = int(counts.get("development", 0))
    adequacy = dev_groups >= 6 and final_groups >= 4
    manifests = {}
    if adequacy:
        for split in ("development", "final"):
            table = clean[clean.split == split].sort_values(["event_id", "issue_time_utc", "object_id"])
            path = output_dir / f"stage8_{split}_object_manifest.csv"; table.to_csv(path, index=False)
            manifests[split] = {"path": path.as_posix(), "rows": len(table), "sha256": sha256_file(path)}
        for path in (output_dir / "stage8_independent_system_groups.csv", output_dir / "stage8_hard_negative_manifest.csv", output_dir / "stage8_rejected_candidates.csv"):
            manifests[path.stem] = {"path": path.as_posix(), "sha256": sha256_file(path)}
    result = {
        "stage": "stage_8_event_qualification_only", "gate_designed": False, "model_predictions_generated": False,
        "development_independent_positive_systems": dev_groups, "final_independent_positive_systems": final_groups,
        "development_clean_rows": int((clean.split == "development").sum()), "final_clean_rows": int((clean.split == "final").sum()),
        "development_unique_issue_times": int(clean.loc[clean.split == "development", "issue_time_utc"].nunique()),
        "final_unique_issue_times": int(clean.loc[clean.split == "final", "issue_time_utc"].nunique()),
        "hard_negative_rows": len(hard), "adequacy_passed": adequacy,
        "authorized_next_step": "gate_predeclaration_only" if adequacy else "predeclare_more_events_before_gate_work",
        "manifests": manifests,
        "configuration_sha256": sha256_file(config_path),
    }
    (output_dir / "stage8_qualification_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/stage_8_qualification.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_8/qualification"))
    args = parser.parse_args(); print(json.dumps(run(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
