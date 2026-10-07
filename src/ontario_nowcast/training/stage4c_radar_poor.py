"""Freeze RADAR_POOR_INITIATION_V2 before any Stage 4C model scoring."""

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
BOUNDARY_PIXELS = 9  # 18 km on the 2 km grid


def run(output_dir: Path) -> dict[str, object]:
    source_manifest = Path("artifacts/stage_4/gate/stage4_final_initiation_holdout_manifest.csv")
    tensors = pd.read_csv("artifacts/stage_4b/final_holdout/exact_holdout_tensor_manifest.csv")
    objects = pd.read_csv(source_manifest)
    objects = objects[objects.clean_pre_radar_initiation.astype(str).str.lower().eq("true")]
    metadata = objects.set_index(["source_event_id", "object_id"])
    rows = []
    for index, record in enumerate(tensors.itertuples(index=False), 1):
        payload = _load_npz(Path(record.tensor_path))
        radar, radar_mask = payload["radar"], payload["radar_mask"] > 0
        target, target_mask = payload["target"], payload["target_mask"] > 0
        eventual = np.any((target >= RAIN) & target_mask, axis=0) & (radar[-1] < RAIN)
        valid_coverage = float(min(radar_mask.mean(), target_mask.mean()))
        final_dry_fraction = float(np.mean(radar[-1][eventual] < RAIN)) if eventual.any() else 0.0
        history_wet_fraction = float(np.mean((radar >= RAIN) & radar_mask))
        pysteps = pysteps_deterministic_extrapolation(radar[-3:], 20)
        pysteps_fraction = (
            float(np.mean(np.any(pysteps[:, eventual] >= RAIN, axis=0)))
            if eventual.any()
            else 1.0
        )
        yy, xx = np.where(eventual)
        boundary_clear = bool(
            eventual.any()
            and yy.min() >= BOUNDARY_PIXELS
            and xx.min() >= BOUNDARY_PIXELS
            and yy.max() < 128 - BOUNDARY_PIXELS
            and xx.max() < 128 - BOUNDARY_PIXELS
        )
        object_row = metadata.loc[(record.event_id, record.object_id)]
        advective = str(object_row.advective_entry_like).lower() == "true"
        qualifies = bool(
            valid_coverage >= 0.95
            and final_dry_fraction >= 0.95
            and history_wet_fraction <= 0.02
            and pysteps_fraction <= 0.20
            and not advective
            and boundary_clear
        )
        rows.append(
            {
                "identity": record.identity,
                "event_id": record.event_id,
                "object_id": record.object_id,
                "issue_time_utc": record.issue_time_utc,
                "valid_coverage_fraction": valid_coverage,
                "final_dry_fraction_at_eventual_pixels": final_dry_fraction,
                "history_wet_area_fraction": history_wet_fraction,
                "pysteps_coverage_at_eventual_pixels": pysteps_fraction,
                "advective_entry_like": advective,
                "boundary_clear_18km": boundary_clear,
                "radar_poor_initiation_v2": qualifies,
            }
        )
        print(f"[radar-poor-v2] {index}/{len(tensors)}", flush=True)
    table = pd.DataFrame(rows)
    selected = table[table.radar_poor_initiation_v2]
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "radar_poor_initiation_v2_membership.csv", index=False)
    by_event = table.groupby("event_id", as_index=False).agg(
        object_rows=("identity", "size"),
        qualifying_rows=("radar_poor_initiation_v2", "sum"),
    )
    by_event.to_csv(output_dir / "radar_poor_initiation_v2_counts_by_event.csv", index=False)
    definition = {
        "name": "RADAR_POOR_INITIATION_V2",
        "frozen_before_stage_c_predictions": True,
        "criteria": {
            "minimum_valid_radar_coverage": 0.95,
            "minimum_final_dry_fraction_at_eventual_pixels": 0.95,
            "maximum_history_wet_area_fraction": 0.02,
            "maximum_pysteps_coverage_at_eventual_pixels": 0.20,
            "advective_entry_like": False,
            "minimum_boundary_distance_km": 18,
        },
        "source_role": "historical Stage 4B holdout; membership only, never Stage C selection",
        "qualifying_object_rows": len(selected),
        "unique_issue_times": int(selected.issue_time_utc.nunique()),
        "independent_weather_events": int(selected.event_id.nunique()),
        "counts_by_event": dict(zip(by_event.event_id, by_event.qualifying_rows.astype(int), strict=True)),
        "membership_sha256": hashlib.sha256(
            (output_dir / "radar_poor_initiation_v2_membership.csv").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "radar_poor_initiation_v2_frozen_definition.json").write_text(
        json.dumps(definition, indent=2), encoding="utf-8"
    )
    return definition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4c/radar_poor_v2"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
