"""Apply the predeclared Stage 4C DEV decision rule to existing score artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def run() -> dict[str, object]:
    root = Path("artifacts/stage_4c/c1/evaluation")
    lifecycle = pd.read_csv(root / "dev_lifecycle_event_macro.csv")
    events = pd.read_csv(root / "dev_lifecycle_by_event.csv")
    hard = pd.read_csv(root / "dev_hard_negative_by_event.csv")
    hard_macro = hard.groupby(["condition", "probability_threshold"], as_index=False).mean(
        numeric_only=True
    )
    candidates = lifecycle.merge(
        hard_macro, on=["condition", "probability_threshold"], suffixes=("", "_hard")
    )
    candidates["constraints_pass"] = (candidates.wet_area_fraction <= 0.01) & (
        candidates.false_initiation_fraction <= 0.40
    )
    normal = candidates[(candidates.condition == "normal") & candidates.constraints_pass]
    selected = normal.sort_values(
        ["f1", "onset_mae_minutes", "brier_score"], ascending=[False, True, True]
    ).iloc[0]
    selected_threshold = float(selected.probability_threshold)

    stage4b = pd.read_csv(
        "artifacts/stage_4b/final_dev_decision/dev_initiation_metrics_by_event.csv"
    )
    comparator_rows = []
    for model, threshold in (("A_plus", 0.35), ("B2", 0.20)):
        frame = stage4b[(stage4b.model == model) & (stage4b.threshold == threshold)]
        row = frame.mean(numeric_only=True).to_dict()
        comparator_rows.append({"model": model, "threshold": threshold, **row})
    c1_events = events[
        (events.condition == "normal") & (events.probability_threshold == selected_threshold)
    ]
    c1 = c1_events.mean(numeric_only=True).to_dict()
    comparator_rows.append({"model": "C1", "threshold": selected_threshold, **c1})
    comparators = pd.DataFrame(comparator_rows)
    comparators.to_csv(root / "dev_lifecycle_selected_comparison.csv", index=False)

    a_events = stage4b[(stage4b.model == "A_plus") & (stage4b.threshold == 0.35)][
        ["event_id", "f1"]
    ].rename(columns={"f1": "a_plus_f1"})
    paired = c1_events[["event_id", "f1"]].merge(a_events, on="event_id")
    deltas = (paired.f1 - paired.a_plus_f1).to_numpy()
    rng = np.random.default_rng(314159)
    bootstrap = np.asarray(
        [np.mean(rng.choice(deltas, len(deltas), replace=True)) for _ in range(10000)]
    )

    broad_c1 = pd.read_csv(root / "dev_broad_metrics.csv")
    a_plus = pd.read_csv("artifacts/stage_4b/a_plus_radar/evaluation_pysteps/dev_metrics.csv")
    b2 = pd.read_csv("artifacts/stage_4b/b2_c13_cooling/evaluation_pysteps/dev_metrics.csv")
    baseline = pd.read_csv("artifacts/stage_4c/baselines/dev_baseline_metrics.csv")
    tables = [broad_c1[broad_c1.condition == "normal"].assign(model="C1")]
    for frame, source, name in (
        (a_plus, "a_plus_radar", "A_plus"),
        (b2, "normal_goes", "B2"),
        (baseline, "pysteps", "PySTEPS"),
        (baseline, "raw_hrrr_apcp", "raw_HRRR_APCP"),
    ):
        tables.append(
            frame[(frame.model == source) & frame.lead_minutes.isin((30, 60, 90, 120))].assign(
                model=name
            )
        )
    columns = [
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
    broad = pd.concat(tables, ignore_index=True).rename(
        columns={"fss_radius_18km": "fss_radius_18km"}
    )
    broad[columns].to_csv(root / "dev_broad_model_comparison.csv", index=False)

    perturb = events[events.probability_threshold == selected_threshold]
    perturb_macro = perturb.groupby("condition", as_index=False).mean(numeric_only=True)
    perturb_macro.to_csv(root / "dev_perturbation_selected_threshold.csv", index=False)
    normal_events = perturb[perturb.condition == "normal"].set_index("event_id")
    dependence = {}
    for condition in ("zeroed", "shuffled_global", "shuffled_within_event"):
        other = perturb[perturb.condition == condition].set_index("event_id")
        event_delta = normal_events.f1 - other.f1
        dependence[condition] = {
            "event_macro_f1_delta": float(event_delta.mean()),
            "events_normal_better": int((event_delta > 0).sum()),
            "events_compared": len(event_delta),
        }
    # Under the frozen qualitative rule, microscopic sign changes are not
    # "meaningful" degradation when inputs are destroyed.
    attribution_pass = all(
        value["event_macro_f1_delta"] >= 0.005 and value["events_normal_better"] >= 2
        for value in dependence.values()
    )
    a_macro = comparators[comparators.model == "A_plus"].iloc[0]
    c_macro = comparators[comparators.model == "C1"].iloc[0]
    incremental_f1 = float(c_macro.f1 - a_macro.f1)
    classification = "NULL" if not attribution_pass and incremental_f1 <= 0 else "MIXED"
    result = {
        "classification": classification,
        "selected_c1_probability_threshold": selected_threshold,
        "constraints": {
            "hard_negative_wet_area_fraction": float(selected.wet_area_fraction),
            "hard_negative_wet_area_fraction_max": 0.01,
            "false_initiation_fraction": float(selected.false_initiation_fraction),
            "false_initiation_fraction_max": 0.40,
            "passed": bool(selected.constraints_pass),
        },
        "c1_event_macro": {
            key: float(c_macro[key])
            for key in (
                "f1",
                "precision",
                "recall",
                "false_initiation_fraction",
                "onset_mae_minutes",
                "brier_score",
            )
        },
        "c1_minus_a_plus_event_macro_f1": incremental_f1,
        "c1_minus_a_plus_event_bootstrap_95pct_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_unit": "weather event",
        "bootstrap_events": len(deltas),
        "thermodynamic_dependence": dependence,
        "meaningful_dependence_floor_event_macro_f1": 0.005,
        "thermodynamic_attribution_passed": attribution_pass,
        "radar_limited_dev_status": (
            "not_scoreable_on_the_common_24_anchor_DEV_set: the frozen historical cohort "
            "contains 14 exact object/time rows from a different diagnostic set, and no "
            "exact C1 thermodynamic tensors exist for those identities; membership was not altered"
        ),
        "subablation_training_authorized": False,
        "protected_2023_holdout_scored": False,
        "stage_c_procedure_frozen": False,
        "next_action": "document the DEV null; do not train C-MOISTURE/C-INSTABILITY or score holdout",
    }
    (root / "dev_decision.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
