"""Freeze the primary RADAR_LIMITED_INITIATION cohort before C1 training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.baselines import pysteps_deterministic_extrapolation
from .stage4b_train import _load_npz

RAIN = 0.1
BOUNDARY_PIXELS = 9


def run(output_dir: Path) -> dict[str, object]:
    objects = pd.read_csv("artifacts/stage_4/gate/event_object_validation.csv")
    objects = objects[objects.clean_pre_radar_initiation.astype(str).str.lower().eq("true")]
    metadata = objects.set_index(["source_event_id", "object_id"])
    tensors = pd.read_csv("artifacts/stage_4b/final_holdout/exact_holdout_tensor_manifest.csv")
    rows = []
    for index, record in enumerate(tensors.itertuples(index=False), 1):
        payload = _load_npz(Path(record.tensor_path))
        radar, radar_mask = payload["radar"], payload["radar_mask"] > 0
        target, target_mask = payload["target"], payload["target_mask"] > 0
        eventual = np.any((target >= RAIN) & target_mask, axis=0) & (radar[-1] < RAIN)
        valid_coverage = float(min(radar_mask.mean(), target_mask.mean()))
        final_dry = float(np.mean(radar[-1][eventual] < RAIN)) if eventual.any() else 0.0
        history_wet = float(np.mean((radar >= RAIN) & radar_mask))
        pysteps = pysteps_deterministic_extrapolation(radar[-3:], 20)
        pysteps_coverage = (
            float(np.mean(np.any(pysteps[:, eventual] >= RAIN, axis=0)))
            if eventual.any() else 1.0
        )
        object_row = metadata.loc[(record.event_id, record.object_id)]
        boundary_clear = bool(
            float(object_row.y_min) - float(object_row.tile_y_min) >= BOUNDARY_PIXELS
            and float(object_row.x_min) - float(object_row.tile_x_min) >= BOUNDARY_PIXELS
            and float(object_row.tile_y_max) - float(object_row.y_max) >= BOUNDARY_PIXELS
            and float(object_row.tile_x_max) - float(object_row.x_max) >= BOUNDARY_PIXELS
        )
        advective = str(object_row.advective_entry_like).lower() == "true"
        qualifies = bool(
            valid_coverage >= 0.95 and final_dry >= 0.90 and history_wet <= 0.05
            and pysteps_coverage <= 0.40 and not advective and boundary_clear
        )
        rows.append({
            "identity": record.identity, "event_id": record.event_id,
            "object_id": record.object_id, "issue_time_utc": record.issue_time_utc,
            "valid_coverage_fraction": valid_coverage,
            "final_dry_fraction_at_eventual_pixels": final_dry,
            "history_wet_area_fraction": history_wet,
            "pysteps_coverage_at_eventual_pixels": pysteps_coverage,
            "advective_entry_like": advective, "boundary_clear_18km": boundary_clear,
            "radar_limited_initiation": qualifies,
        })
        print(f"[radar-limited] {index}/{len(tensors)}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    path = output_dir / "radar_limited_initiation_membership.csv"
    table.to_csv(path, index=False)
    selected = table[table.radar_limited_initiation]
    by_event = table.groupby("event_id", as_index=False).agg(
        object_rows=("identity", "size"), qualifying_rows=("radar_limited_initiation", "sum")
    )
    by_event.to_csv(output_dir / "radar_limited_initiation_counts_by_event.csv", index=False)
    result = {
        "name": "RADAR_LIMITED_INITIATION",
        "role": "primary radar-poor diagnostic",
        "frozen_before_c1_training_or_scoring": True,
        "criteria": {
            "minimum_valid_radar_coverage": 0.95,
            "minimum_final_dry_fraction_at_eventual_pixels": 0.90,
            "maximum_history_wet_area_fraction": 0.05,
            "maximum_pysteps_coverage_at_eventual_pixels": 0.40,
            "advective_entry_like": False,
            "minimum_boundary_distance_km": 18,
        },
        "threshold_adjustments_after_initial_proposal": "none; boundary criterion corrected to use the tracked initiation footprint rather than unrelated future rain anywhere in the tile",
        "historical_object_rows": len(selected),
        "historical_unique_issue_times": int(selected.issue_time_utc.nunique()),
        "historical_independent_events": int(selected.event_id.nunique()),
        "membership_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output_dir / "radar_limited_initiation_frozen_definition.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4c/radar_limited"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
