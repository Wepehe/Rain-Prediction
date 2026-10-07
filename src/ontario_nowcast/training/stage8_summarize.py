"""Package frozen Stage 8 nested-development outputs without refitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.manifest import sha256_file


ROOT = Path("artifacts/stage_8")
OUT = ROOT / "nested_development"
PREDECL = "18e2d0273f5e3a04028339e5cd534d9b2e86a3a8129d32234aaa252e7e94ec5d"
METRICS = [
    "precision", "recall", "f1", "false_initiation_fraction",
    "onset_mae_minutes", "median_onset_bias_minutes", "brier_score",
    "detection_within_30min", "detection_within_60min",
    "detection_within_90min", "detection_within_120min",
]


def main() -> None:
    manifest = pd.read_csv(ROOT / "development_features/development_feature_manifest.csv")
    scores = pd.read_csv(OUT / "outer_row_metrics.csv")
    events = pd.read_csv(OUT / "comparator_by_event_metrics.csv")
    behavior = pd.read_csv(OUT / "gate_behavior.csv")
    coefficients = pd.read_csv(OUT / "outer_coefficients.csv")
    fits = json.loads((OUT / "outer_fold_fits.json").read_text())

    positive = scores[scores.role.eq("positive_initiation")]
    positive.groupby("model", as_index=False)[METRICS].mean(numeric_only=True).to_csv(
        OUT / "comparator_row_macro_metrics.csv", index=False
    )

    stage8 = events[events.model.eq("Stage8_gate")].set_index("independent_system_group")
    aplus = events[events.model.eq("A_plus")].set_index("independent_system_group")
    hybrid = events[events.model.eq("Stage6_hybrid")].set_index("independent_system_group")
    thresholds = {x["outer_system"]: x["threshold"] for x in fits}
    event_table = pd.DataFrame({
        "independent_system_group": stage8.index,
        "a_plus_f1": aplus.loc[stage8.index, "f1"].values,
        "hybrid_f1": hybrid.loc[stage8.index, "f1"].values,
        "stage8_f1": stage8.f1.values,
        "stage8_minus_a_plus": (stage8.f1 - aplus.loc[stage8.index, "f1"]).values,
        "stage8_minus_hybrid": (stage8.f1 - hybrid.loc[stage8.index, "f1"]).values,
        "selected_threshold": [thresholds[x] for x in stage8.index],
        "mean_gate_value": [behavior.loc[behavior.event.eq(x), "g"].mean() for x in stage8.index],
    })
    event_table.to_csv(OUT / "stage8_individual_system_results.csv", index=False)

    quantiles = behavior.g.quantile([0, .05, .25, .5, .75, .95, 1]).rename_axis("quantile").reset_index(name="g")
    quantiles.to_csv(OUT / "gate_quantiles.csv", index=False)
    behavior.groupby("lead_minutes").g.agg(["mean", "median"]).reset_index().to_csv(OUT / "gate_by_lead.csv", index=False)
    behavior.groupby("event").g.agg(["mean", "median"]).reset_index().to_csv(OUT / "gate_by_event.csv", index=False)
    behavior.assign(confidence_bin=pd.cut(behavior.a_plus_confidence, [0, .25, .5, .75, 1], include_lowest=True)).groupby(
        "confidence_bin", observed=True, as_index=False
    ).g.agg(["count", "mean", "median"]).reset_index(drop=True).to_csv(OUT / "gate_by_a_plus_confidence.csv", index=False)
    behavior.assign(agreement_bin=pd.cut(behavior.agreement, [0, .5, .75, .9, 1], include_lowest=True)).groupby(
        "agreement_bin", observed=True, as_index=False
    ).g.agg(["count", "mean", "median"]).reset_index(drop=True).to_csv(OUT / "gate_by_agreement.csv", index=False)
    behavior.assign(hrrr_state=np.where(behavior.hrrr_wet_fraction.gt(0), "wet", "dry")).groupby(
        "hrrr_state", as_index=False
    ).g.agg(["count", "mean", "median"]).reset_index(drop=True).to_csv(OUT / "gate_by_hrrr_state.csv", index=False)

    cohorts = pd.read_csv(ROOT / "development_features/stage8_development_radar_cohorts.csv")
    limited = set(cohorts.loc[cohorts.radar_limited_initiation, "row_id"])
    v2 = set(cohorts.loc[cohorts.radar_poor_initiation_v2, "row_id"])
    subsets = []
    for name, mask in {
        "all": np.ones(len(behavior), dtype=bool),
        "hard_negative": behavior.role.eq("hard_negative"),
        "RADAR_LIMITED_INITIATION": behavior.row_id.isin(limited),
        "RADAR_POOR_INITIATION_V2": behavior.row_id.isin(v2),
    }.items():
        values = behavior.loc[mask, "g"]
        subsets.append({"subset": name, "count": len(values), "mean": values.mean(), "median": values.median(), "q05": values.quantile(.05), "q95": values.quantile(.95)})
    pd.DataFrame(subsets).to_csv(OUT / "gate_by_mechanism_subset.csv", index=False)

    intercepts = pd.DataFrame([{"outer_system": x["outer_system"], "feature": "intercept", "coefficient": x["intercept"]} for x in fits])
    coefficients = pd.concat([coefficients, intercepts], ignore_index=True)
    coefficients.to_csv(OUT / "outer_coefficients_with_intercept.csv", index=False)
    coefficients.groupby("feature").coefficient.agg(["mean", "std", "min", "max"]).reset_index().to_csv(
        OUT / "coefficient_stability.csv", index=False
    )

    hard = pd.read_csv(OUT / "hard_negative_outer_metrics.csv")
    pd.DataFrame([{
        "rows": len(hard),
        "systems": hard.independent_system_group.nunique(),
        "wet_area_fraction": hard.wet_area_fraction.mean(),
        "false_initiation_fraction": hard.false_initiation_fraction.mean(),
        "brier_score": hard.brier_score.mean(),
        "mean_probability": hard.mean_probability.mean(),
        "maximum_probability": hard.maximum_probability.max(),
        "mean_gate_value": hard.mean_gate_value.mean(),
    }]).to_csv(OUT / "hard_negative_summary.csv", index=False)

    predicted_rows = []
    files_valid = True
    categorical_present = True
    for fold in sorted(OUT.glob("fold_*")):
        pm = pd.read_csv(fold / "prediction_manifest.csv")
        predicted_rows.extend(pm.row_id.tolist())
        for row in pm.itertuples(index=False):
            files_valid &= sha256_file(Path(row.prediction_path)) == row.sha256
            with np.load(row.prediction_path) as z:
                categorical_present &= "prediction" in z.files
    audit = {
        "predeclaration_sha256": sha256_file(ROOT / "gate_predeclaration/stage8_gate_predeclaration.json"),
        "predeclaration_matches_authoritative": sha256_file(ROOT / "gate_predeclaration/stage8_gate_predeclaration.json") == PREDECL,
        "development_rows": len(manifest),
        "positive_rows": int(manifest.role.eq("positive_initiation").sum()),
        "hard_negative_rows": int(manifest.role.eq("hard_negative").sum()),
        "unique_predicted_rows": len(set(predicted_rows)),
        "each_row_predicted_exactly_once": len(predicted_rows) == len(set(predicted_rows)) == len(manifest),
        "prediction_hashes_valid": bool(files_valid),
        "probabilities_and_categorical_predictions_stored": bool(categorical_present),
        "all_outer_optimizers_converged": all(x["optimizer_success"] for x in fits),
        "final_set_scored": False,
    }
    (OUT / "nested_execution_integrity.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
