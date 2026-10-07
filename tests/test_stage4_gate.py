import json
from pathlib import Path

import pandas as pd

from ontario_nowcast.training.stage4_gate import (
    freeze_final_holdout_manifest,
    summarize_hard_negative_decisions,
)


def test_final_holdout_manifest_is_metadata_only_and_hashed(tmp_path: Path) -> None:
    objects = pd.DataFrame(
        {
            "source_event_id": ["holdout_event", "dev_event"],
            "object_id": ["initiation_track_0001", "initiation_track_0001"],
            "split": ["stage4_initiation_holdout", "stage4_dev_repair"],
            "event_class": ["positive_initiation", "positive_initiation"],
            "representative_issue_time_utc": [
                "2026-07-27T18:00:00+00:00",
                "2026-06-18T18:00:00+00:00",
            ],
            "tile_y_min": [10, 20],
            "tile_y_max": [137, 147],
            "tile_x_min": [30, 40],
            "tile_x_max": [157, 167],
            "tile_bbox_wgs84": ["-84,42,-82,44", "-83,42,-81,44"],
            "clean_pre_radar_initiation": [True, True],
            "advective_entry_like": [False, False],
            "paired_negative_id": ["dry_pair", ""],
        }
    )

    metadata = freeze_final_holdout_manifest(objects, tmp_path)

    assert metadata["rows"] == 1
    assert metadata["clean_pre_radar_initiation_rows"] == 1
    assert metadata["sha256"]
    manifest = pd.read_csv(tmp_path / "stage4_final_initiation_holdout_manifest.csv")
    assert manifest["source_event_id"].tolist() == ["holdout_event"]
    sidecar = json.loads(
        (tmp_path / "stage4_final_initiation_holdout_manifest.sha256.json").read_text()
    )
    assert sidecar["scoring_policy"].startswith("metadata-only freeze")


def test_hard_negative_decisions_block_gate_when_any_pair_is_rejected(tmp_path: Path) -> None:
    verifier_output = tmp_path / "hard_negative_pair_matching.csv"
    pd.DataFrame(
        {
            "positive_event_id": ["positive_a", "positive_b"],
            "negative_event_id": ["negative_a", "negative_b"],
            "verified_hard_negative": [True, False],
            "rejection_reason": ["", "radar_precipitation_too_extensive"],
        }
    ).to_csv(verifier_output, index=False)

    _, summary = summarize_hard_negative_decisions(verifier_output, tmp_path)

    assert summary["hard_negative_gate_passed"] is False
    assert summary["pairs_attempted"] == 2
    assert summary["pairs_verified"] == 1
    assert summary["pairs_rejected"] == 1
    assert summary["rejected_negative_event_ids"] == ["negative_b"]
