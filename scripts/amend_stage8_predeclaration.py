"""Apply the final pre-training Stage 8 grouped-validation amendment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("artifacts/stage_8")
SPEC_PATH = ROOT / "gate_predeclaration/stage8_gate_predeclaration.json"
EXPECTED_PRE_AMENDMENT_SHA256 = "10fef144820a4f63ad4ba1f5515adedfeb979f1fe238fdce3cfe63f2f5e109c9"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha(SPEC_PATH) != EXPECTED_PRE_AMENDMENT_SHA256:
        raise RuntimeError("Stage 8 predeclaration changed before the amendment")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec["gate_trained"] or spec["development_predictions_generated"]:
        raise RuntimeError("amendment is allowed only before any Stage 8 fit or prediction")

    positive = pd.read_csv(ROOT / "qualification/stage8_development_object_manifest.csv")
    negative = pd.read_csv(ROOT / "qualification/stage8_hard_negative_manifest.csv")
    negative = negative[negative["split"].eq("development")].copy()
    positive_groups = sorted(positive["independent_system_group"].unique())
    negative_groups = sorted(negative["independent_system_group"].unique())
    additional_negative_groups = sorted(set(negative_groups) - set(positive_groups))
    if len(positive_groups) != 7 or additional_negative_groups:
        raise RuntimeError("membership audit does not support the frozen seven-fold amendment")

    rows: list[dict[str, str]] = []
    for row in positive.sort_values(["independent_system_group", "event_id", "issue_time_utc", "object_id"]).itertuples(index=False):
        rows.append({
            "row_id": str(row.object_id),
            "role": "positive_initiation",
            "candidate_event_id": str(row.event_id),
            "independent_system_group": str(row.independent_system_group),
            "relationship_to_positive_systems": "member_of_qualified_positive_system",
        })
    for row in negative.sort_values(["independent_system_group", "event_id", "issue_time_utc", "hard_negative_id"]).itertuples(index=False):
        rows.append({
            "row_id": str(row.hard_negative_id),
            "role": "hard_negative",
            "candidate_event_id": str(row.event_id),
            "independent_system_group": str(row.independent_system_group),
            "relationship_to_positive_systems": "same_predeclared_candidate_period_and_system_as_qualified_positive_rows",
        })
    membership = pd.DataFrame(rows)
    membership_path = ROOT / "gate_predeclaration/stage8_development_row_system_membership.csv"
    membership.to_csv(membership_path, index=False)

    folds = []
    for group in positive_groups:
        training = [item for item in positive_groups if item != group]
        hard_ids = membership.loc[
            membership["role"].eq("hard_negative")
            & membership["independent_system_group"].eq(group), "row_id"
        ].tolist()
        folds.append({
            "outer_fold_id": f"holdout__{group}",
            "outer_held_out_system": group,
            "outer_training_systems": training,
            "outer_held_out_positive_rows": int(((membership.role == "positive_initiation") & (membership.independent_system_group == group)).sum()),
            "outer_held_out_hard_negative_row_ids": hard_ids,
            "inner_folds": [
                {"inner_held_out_system": inner, "inner_training_systems": [item for item in training if item != inner]}
                for inner in training
            ],
            "threshold_source": "combined inner grouped out-of-system predictions from the six outer-training systems only",
        })

    spec["status"] = "PREDECLARED_AMENDED_NOT_TRAINED"
    spec["pretraining_amendment"] = {
        "purpose": "make promotion estimation and hard-negative constraints genuinely out-of-system",
        "made_before_any_stage8_coefficient_fit": True,
        "stage8_predictions_existed_at_amendment": False,
        "final_forecast_outputs_existed_at_amendment": False,
        "consumed_2023_consulted": False,
        "unchanged": [
            "eight features", "logistic architecture", "alpha 0.50", "HRRR occurrence definition",
            "A+ checkpoint", "lambda 1.0", "L-BFGS solver", "event-balanced Brier objective",
            "comparators", "radar-limited definitions", "final object manifest"
        ],
    }
    spec["development_row_system_membership"] = {
        "path": membership_path.as_posix(),
        "sha256": sha(membership_path),
        "rows": len(membership),
        "positive_rows": int((membership.role == "positive_initiation").sum()),
        "hard_negative_rows": int((membership.role == "hard_negative").sum()),
        "positive_system_groups": positive_groups,
        "hard_negative_system_groups": negative_groups,
        "additional_negative_only_system_groups": additional_negative_groups,
        "all_hard_negative_systems_are_existing_positive_groups": True,
        "row_memberships": rows,
    }
    spec["development_validation"] = {
        "method": "nested leave-one-independent-weather-system-out promotion evaluation",
        "universal_group_policy": "every positive and hard-negative row is held out from any fit that evaluates that row",
        "outer_positive_system_folds": 7,
        "outer_folds": folds,
        "outer_training_systems_per_fold": 6,
        "outer_held_out_positive_systems_per_fold": 1,
        "same_system_hard_negatives_follow_outer_holdout": True,
        "additional_negative_only_outer_folds": 0,
        "inner_policy": "within each outer training partition, perform six-fold grouped LOEO prediction; each inner validation system is absent from its inner fit",
        "inner_normalization": "fit on inner-training systems only",
        "outer_normalization": "fit on the six outer-training systems only",
        "outer_coefficients": "fit on the six outer-training systems only after inner threshold selection",
        "september_7_8_same_group": True,
        "promotion_predictions": "only predictions made by the outer model for its untouched outer system, using that fold's inner-selected threshold",
        "prohibited_promotion_inputs": [
            "in-sample predictions", "inner-CV predictions", "outer-event-informed threshold",
            "globally selected all-seven-system threshold"
        ],
    }
    frozen_rule = {
        "grid": [round(value / 100, 2) for value in range(5, 100, 5)],
        "eligibility": {"hard_negative_wet_area_max": 0.01, "initiation_false_fraction_max": 0.40},
        "maximize": "unweighted weather-event-macro initiation F1",
        "tie_breakers_in_order": ["lower onset MAE", "lower Brier", "higher precision", "lower threshold"],
    }
    spec["promotion_threshold_selection"] = {
        **frozen_rule,
        "scope": "separate inner grouped predictions within each outer fold",
        "selected_threshold_count": 7,
        "outer_system_outcomes_used": False,
        "role": "development promotion estimation only",
    }
    spec["final_fit_threshold_selection"] = {
        **frozen_rule,
        "scope": "ordinary all-development seven-system OOF probabilities, only after nested promotion passes",
        "selected_threshold_count": 1,
        "role": "final model configuration, not development promotion estimation",
        "final_holdout_used": False,
    }
    spec.pop("operating_threshold_selection", None)
    spec["hard_negative_out_of_system_evaluation"] = {
        "systems": len(negative_groups),
        "rows": len(negative),
        "additional_negative_only_systems": 0,
        "policy": "use the outer model and inner-selected threshold for the coincident positive-system fold",
        "row_mapping": [
            {
                "hard_negative_row_id": row["row_id"],
                "independent_system_group": row["independent_system_group"],
                "outer_model_fit": f"holdout__{row['independent_system_group']}",
                "training_systems": [item for item in positive_groups if item != row["independent_system_group"]],
                "threshold_source": f"inner_grouped_predictions_within__holdout__{row['independent_system_group']}",
            }
            for row in rows if row["role"] == "hard_negative"
        ],
    }
    spec["promotion"]["primary_endpoint"] = "unweighted weather-event-macro initiation F1 from the seven untouched nested outer positive-system predictions"
    spec["promotion"]["conditions"]["system_consistency"] = "at least 4 of 7 independent positive systems have strictly positive Stage8-minus-A+ event F1; ties do not count"
    spec["promotion"]["conditions"]["hard_negative_wet_area"] = "out-of-system hard-negative wet area <= 0.01"
    spec["promotion"]["conditions"]["false_initiation_fraction"] = "outer-evaluation false-initiation fraction <= 0.40"
    spec["promotion"]["prediction_source"] = "combined untouched nested outer predictions only"
    spec["post_promotion_final_fit"] = {
        "authorized_only_if_all_nested_promotion_conditions_pass": True,
        "steps_in_order": [
            "select one deployment threshold using ordinary all-seven-system OOF probabilities and the frozen rule",
            "calculate normalization from all seven development systems",
            "fit one logistic gate on all seven development systems",
            "freeze normalization, coefficients, deployment threshold, code/source hashes, comparators, and cohort memberships",
            "write the complete final Stage 8 procedure manifest",
            "only then score the sealed six-system final holdout once"
        ],
        "ordinary_oof_threshold_role": "final model configuration only",
        "nested_outer_threshold_role": "development promotion estimate only",
    }

    SPEC_PATH.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    digest = sha(SPEC_PATH)
    (SPEC_PATH.parent / "stage8_gate_predeclaration.sha256").write_text(
        f"{digest}  {SPEC_PATH.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "predeclaration_sha256": digest,
        "membership_sha256": sha(membership_path),
        "membership_rows": len(membership),
        "positive_systems": len(positive_groups),
        "hard_negative_systems": len(negative_groups),
        "additional_negative_systems": len(additional_negative_groups),
        "gate_trained": spec["gate_trained"],
    }, indent=2))


if __name__ == "__main__":
    main()
