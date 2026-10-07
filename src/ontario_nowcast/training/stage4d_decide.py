"""Apply the immutable Stage 4D DEV promotion rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def run() -> dict[str, object]:
    root = Path("artifacts/stage_4d/d1/evaluation")
    macro = pd.read_csv(root / "dev_lifecycle_event_macro.csv")
    events = pd.read_csv(root / "dev_lifecycle_by_event.csv")
    negatives = pd.read_csv(root / "dev_hard_negative_by_event.csv")
    negative_macro = negatives.groupby(["condition", "probability_threshold"], as_index=False).mean(
        numeric_only=True
    )
    candidates = macro.merge(
        negative_macro, on=["condition", "probability_threshold"], suffixes=("", "_hard")
    )
    normal = candidates[
        (candidates.condition == "normal")
        & (candidates.wet_area_fraction <= 0.01)
        & (candidates.false_initiation_fraction <= 0.40)
    ]
    selected = normal.sort_values(
        ["f1", "onset_mae_minutes", "brier_score"], ascending=[False, True, True]
    ).iloc[0]
    threshold = float(selected.probability_threshold)
    selected_conditions = candidates[candidates.probability_threshold.eq(threshold)].copy()
    selected_conditions.to_csv(root / "dev_selected_condition_metrics.csv", index=False)

    prior_sweep = pd.read_csv(
        "artifacts/stage_4b/final_dev_decision/dev_common_probability_policy_sweep.csv"
    )
    prior_events = pd.read_csv(
        "artifacts/stage_4b/final_dev_decision/dev_initiation_metrics_by_event.csv"
    )
    comparator_rows = []
    for model, value in (("A_plus", 0.35), ("B2", 0.20)):
        row = prior_sweep[
            (prior_sweep.model == model) & prior_sweep.probability_threshold.eq(value)
        ].iloc[0]
        comparator_rows.append({"model": model, "probability_threshold": value, **row.to_dict()})
    d1 = selected.to_dict()
    comparator_rows.append({"model": "D1", **d1})
    c1 = pd.read_csv("artifacts/stage_4c/c1/evaluation/dev_lifecycle_event_macro.csv")
    c1 = c1[(c1.condition == "normal") & c1.probability_threshold.eq(0.35)].iloc[0]
    comparator_rows.append({"model": "C1", **c1.to_dict()})
    pd.DataFrame(comparator_rows).to_csv(root / "dev_lifecycle_model_comparison.csv", index=False)

    broad_tables = [
        pd.read_csv(root / "dev_broad_metrics.csv")
        .query("condition == 'normal'")
        .assign(model="D1"),
        pd.read_csv("artifacts/stage_4c/c1/evaluation/dev_broad_metrics.csv")
        .query("condition == 'normal'")
        .assign(model="C1"),
    ]
    for path, source, name in (
        (
            "artifacts/stage_4b/a_plus_radar/evaluation_pysteps/dev_metrics.csv",
            "a_plus_radar",
            "A_plus",
        ),
        (
            "artifacts/stage_4b/b2_c13_cooling/evaluation_pysteps/dev_metrics.csv",
            "normal_goes",
            "B2",
        ),
        ("artifacts/stage_4c/baselines/dev_baseline_metrics.csv", "pysteps", "PySTEPS"),
        ("artifacts/stage_4c/baselines/dev_baseline_metrics.csv", "raw_hrrr_apcp", "raw_HRRR_APCP"),
    ):
        frame = pd.read_csv(path)
        broad_tables.append(
            frame[(frame.model == source) & frame.lead_minutes.isin((30, 60, 90, 120))].assign(
                model=name
            )
        )
    broad = pd.concat(broad_tables, ignore_index=True)
    broad[
        [
            "model",
            "lead_minutes",
            "threshold_mm_hr",
            "csi",
            "pod",
            "far",
            "f1",
            "fss_radius_18km",
            "brier_score",
        ]
    ].to_csv(root / "dev_broad_model_comparison.csv", index=False)

    d_events = events[(events.condition == "normal") & events.probability_threshold.eq(threshold)][
        ["event_id", "f1"]
    ]
    a_events = prior_events[(prior_events.model == "A_plus") & prior_events.threshold.eq(0.35)][
        ["event_id", "f1"]
    ].rename(columns={"f1": "a_plus_f1"})
    paired = d_events.merge(a_events, on="event_id")
    paired["d1_minus_a_plus_f1"] = paired.f1 - paired.a_plus_f1
    paired.to_csv(root / "dev_d1_minus_a_plus_by_event.csv", index=False)
    deltas = paired.d1_minus_a_plus_f1.to_numpy()
    rng = np.random.default_rng(271828)
    boot = np.asarray(
        [np.mean(rng.choice(deltas, len(deltas), replace=True)) for _ in range(10000)]
    )

    dependence = {}
    for condition in ("zeroed", "shuffled_global", "shuffled_within_event"):
        row = selected_conditions[selected_conditions.condition == condition].iloc[0]
        dependence[condition] = float(selected.f1 - row.f1)
    physical = {}
    broad_d1 = pd.read_csv(root / "dev_broad_metrics.csv")
    for condition in ("convergence_removed", "vertical_motion_removed", "shear_removed"):
        row = selected_conditions[selected_conditions.condition == condition].iloc[0]
        skill = broad_d1[
            (broad_d1.condition == condition)
            & broad_d1.threshold_mm_hr.eq(0.1)
            & broad_d1.lead_minutes.isin((60, 90, 120))
        ]
        physical[condition] = {
            "event_macro_f1": float(row.f1),
            "precision": float(row.precision),
            "recall": float(row.recall),
            "false_initiation_fraction": float(row.false_initiation_fraction),
            "onset_mae_minutes": float(row.onset_mae_minutes),
            "brier_score": float(row.brier_score),
            "csi_60_90_120": {
                str(int(item.lead_minutes)): float(item.csi) for item in skill.itertuples()
            },
        }

    a = prior_sweep[
        (prior_sweep.model == "A_plus") & prior_sweep.probability_threshold.eq(0.35)
    ].iloc[0]
    b2 = prior_sweep[(prior_sweep.model == "B2") & prior_sweep.probability_threshold.eq(0.20)].iloc[
        0
    ]
    primary_delta = float(selected.f1 - a.f1)
    gates = {
        "A_d1_f1_at_least_a_plus_plus_0p01": primary_delta >= 0.01,
        "B_normal_minus_zeroed_at_least_0p005": dependence["zeroed"] >= 0.005,
        "C_normal_minus_global_shuffle_at_least_0p005": dependence["shuffled_global"] >= 0.005,
        "D_normal_minus_within_event_shuffle_at_least_0p005": dependence["shuffled_within_event"]
        >= 0.005,
        "E_hard_negative_acceptable": bool(
            selected.wet_area_fraction <= 0.01 and selected.false_initiation_fraction <= 0.40
        ),
    }
    c1_negative = pd.read_csv("artifacts/stage_4c/c1/evaluation/dev_hard_negative_by_event.csv")
    c1_negative = c1_negative[
        (c1_negative.condition == "normal") & c1_negative.probability_threshold.eq(0.35)
    ].mean(numeric_only=True)
    pd.DataFrame(
        [
            {
                "model": "A_plus",
                "threshold": 0.35,
                "mean_probability": a.hard_negative_mean_probability,
                "max_probability": a.hard_negative_max_probability,
                "wet_area_fraction": a.hard_negative_fraction_above_threshold,
                "brier_score": a.hard_negative_brier,
            },
            {
                "model": "B2",
                "threshold": 0.20,
                "mean_probability": b2.hard_negative_mean_probability,
                "max_probability": b2.hard_negative_max_probability,
                "wet_area_fraction": b2.hard_negative_fraction_above_threshold,
                "brier_score": b2.hard_negative_brier,
            },
            {
                "model": "C1",
                "threshold": 0.35,
                "mean_probability": c1_negative.mean_probability,
                "max_probability": c1_negative.max_probability,
                "wet_area_fraction": c1_negative.wet_area_fraction,
                "brier_score": c1_negative.brier_score,
            },
            {
                "model": "D1",
                "threshold": threshold,
                "mean_probability": selected.mean_probability,
                "max_probability": selected.max_probability,
                "wet_area_fraction": selected.wet_area_fraction,
                "brier_score": selected.brier_score_hard,
            },
        ]
    ).to_csv(root / "dev_hard_negative_model_comparison.csv", index=False)
    result = {
        "classification": "PROMISING DYNAMICS SIGNAL" if all(gates.values()) else "NULL",
        "selected_probability_threshold": threshold,
        "d1_event_macro": {
            key: float(selected[key])
            for key in (
                "precision",
                "recall",
                "f1",
                "false_initiation_fraction",
                "onset_mae_minutes",
                "median_onset_bias_minutes",
                "brier_score",
                "detection_within_30min",
                "detection_within_60min",
                "detection_within_90min",
                "detection_within_120min",
            )
        },
        "d1_minus_a_plus": {
            "f1": primary_delta,
            "precision": float(selected.precision - a.precision),
            "recall": float(selected.recall - a.recall),
            "false_initiation_fraction": float(
                selected.false_initiation_fraction - a.false_initiation_fraction
            ),
            "onset_mae_minutes": float(selected.onset_mae_minutes - a.onset_mae_minutes),
            "brier_score": float(selected.brier_score - a.brier_score),
            "weather_event_bootstrap_95pct_ci_f1": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
        },
        "d1_minus_b2": {
            key: float(selected[key] - b2[key])
            for key in (
                "precision",
                "recall",
                "f1",
                "false_initiation_fraction",
                "onset_mae_minutes",
                "brier_score",
            )
        },
        "d1_minus_c1": {
            key: float(selected[key] - c1[key])
            for key in (
                "precision",
                "recall",
                "f1",
                "false_initiation_fraction",
                "onset_mae_minutes",
                "brier_score",
            )
        },
        "normal_minus_destroyed_event_macro_f1": dependence,
        "physical_component_removal": physical,
        "promotion_gates": gates,
        "historical_radar_limited": "not_materialized; secondary endpoint cannot alter failed primary gates",
        "protected_2023_holdout_scored": False,
        "stage_d_closed": not all(gates.values()),
        "next_experiment": "architectural/hybrid information-use test"
        if not all(gates.values())
        else "freeze procedure before holdout decision",
    }
    (root / "dev_decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
