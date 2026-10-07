"""Freeze and verify the transparent Stage 6 hybrid before holdout scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("artifacts/stage_6")
HOLDOUT = Path("artifacts/stage_4c/frozen_holdout/stage4c_final_holdout_manifest.csv")
A_PLUS = Path("artifacts/stage_4b/a_plus_radar/model_best.pt")
EXPECTED_HOLDOUT = "bbf001f7512dc5aae36158941affb878ec7f3e1d011dc0fadec39fed755b3f5c"
EXPECTED_A_PLUS = "63e049c5ea75c3aa6af7b461fe6e9bca2db3c51198a667e71f02a7c6217efe70"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def membership_hash(table: pd.DataFrame, column: str) -> str:
    selected = table[table[column].astype(bool)]
    identities = sorted(
        f"{row.source_event_id}|{row.object_id}|{row.representative_issue_time_utc}"
        for row in selected.itertuples(index=False)
    )
    return hashlib.sha256(("\n".join(identities) + "\n").encode()).hexdigest()


def blend(a_plus_probability, hrrr_occurrence):
    return 0.5 * np.asarray(a_plus_probability) + 0.5 * np.asarray(hrrr_occurrence)


def run() -> dict:
    if sha(HOLDOUT) != EXPECTED_HOLDOUT or sha(A_PLUS) != EXPECTED_A_PLUS:
        raise RuntimeError("frozen holdout or A+ checkpoint hash changed")
    table = pd.read_csv(HOLDOUT)
    if len(table) != 68:
        raise RuntimeError("Stage 6 requires the exact frozen 68-row manifest")
    synthetic = {
        "both_dry": float(blend(0.0, 0.0)),
        "a_plus_only": float(blend(1.0, 0.0)),
        "hrrr_only": float(blend(0.0, 1.0)),
        "both_wet": float(blend(1.0, 1.0)),
    }
    if synthetic != {"both_dry": 0.0, "a_plus_only": 0.5, "hrrr_only": 0.5, "both_wet": 1.0}:
        raise RuntimeError("hybrid synthetic identity gate failed")
    ROOT.mkdir(parents=True, exist_ok=True)
    procedure = {
        "stage": "stage_6_transparent_a_plus_hrrr_hybrid_validation",
        "created_before_protected_predictions": True,
        "scientific_question": "Does a low-degree-of-freedom combination of a precise radar-based nowcast and coarse operational HRRR precipitation provide robust incremental initiation skill on independent weather events?",
        "development_history": {
            "structure_predeclared_before_stage5_h1_evaluation": True,
            "candidate_global_hrrr_weights": [0.0, 0.25, 0.5],
            "selection_data": "DEV only",
            "frozen_selected_weight": 0.5,
            "frozen_operating_probability_threshold": 0.3,
            "further_tuning_choices": "none",
            "h1_or_hybrid_previously_scored_on_2023": False,
        },
        "a_plus": {
            "checkpoint": A_PLUS.as_posix(),
            "checkpoint_sha256": EXPECTED_A_PLUS,
            "probability": "sigmoid of frozen P(rate >= 0.1 mm/h) logit",
        },
        "hrrr": {
            "product": "NOAA HRRR wrfsfc APCP surface hourly accumulation",
            "issue_safety": "most recent established cycle with 60-minute availability lag",
            "contexts": {
                "leads_6_to_60_minutes": "f02 1-2h APCP",
                "leads_66_to_120_minutes": "f03 2-3h APCP",
            },
            "occurrence": "binary APCP interval-average rate >= 0.1 mm/h",
            "temporal_caveat": "hourly interval context, not a six-minute forecast",
        },
        "formula": "hybrid_probability = 0.50 * A_plus_probability + 0.50 * binary_HRRR_occurrence",
        "operating_threshold": 0.3,
        "lifecycle": "first 6-minute lead crossing at each pixel; evaluation restricted to pixels dry in the final radar input for initiation rows",
        "metrics": "existing stage4b_holdout _row_metrics; categorical_metrics; FSS at 18 km; event macro is unweighted mean of event metrics",
        "protected_holdout": {"path": HOLDOUT.as_posix(), "sha256": EXPECTED_HOLDOUT, "rows": 68},
        "memberships": {
            "radar_limited_initiation_sha256": membership_hash(table, "radar_limited_initiation"),
            "radar_poor_initiation_v2_sha256": membership_hash(table, "radar_poor_initiation_v2"),
            "radar_limited_rows": int(table.radar_limited_initiation.sum()),
            "radar_poor_v2_rows": int(table.radar_poor_initiation_v2.sum()),
        },
        "interpretation_classes": [
            "ROBUST HYBRID GAIN",
            "MIXED GENERALIZATION",
            "NULL",
            "NEGATIVE",
        ],
        "protected_result_dependent_branching": False,
    }
    path = ROOT / "frozen_procedure_manifest.json"
    path.write_text(json.dumps(procedure, indent=2), encoding="utf-8")
    digest = sha(path)
    gate = {
        "a_plus_hash_verified": True,
        "holdout_hash_verified": True,
        "weight_exactly_0.50": True,
        "threshold_exactly_0.30": True,
        "no_normalization_refit": True,
        "no_calibration_fit": True,
        "synthetic_formula_cases": synthetic,
        "procedure_sha256": digest,
        "predictions_generated": False,
    }
    (ROOT / "preprediction_identity_gate.json").write_text(json.dumps(gate, indent=2))
    return gate


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
