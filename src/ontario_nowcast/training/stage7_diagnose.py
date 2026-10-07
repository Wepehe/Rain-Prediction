"""Descriptive Stage 7 post-mortem of the consumed 2023 holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from ..evaluation.onset import first_crossing_minutes

ROOT = Path("artifacts/stage_7")
RAIN = 0.1
POLICY = {"A_plus": 0.35, "raw_HRRR": 0.5, "hybrid": 0.3}


def _median_onset(field: np.ndarray, threshold: float, mask: np.ndarray) -> float:
    onset = first_crossing_minutes(field[:, mask], threshold, 6)
    return float(np.nanmedian(onset)) if np.isfinite(onset).any() else float("nan")


def _state(probability: np.ndarray, threshold: float, observed_event: np.ndarray) -> str:
    predicted = np.any(probability >= threshold, axis=0)
    hit = bool(np.any(predicted & observed_event))
    false = bool(np.any(predicted & ~observed_event))
    return ("TP" if hit else "FN") + ("+FP" if false else "")


def _spatial(observed: np.ndarray, hrrr: np.ndarray) -> dict[str, float | str]:
    if not observed.any():
        return {"hrrr_spatial_class": "no observed footprint"}
    if not hrrr.any():
        return {
            "hrrr_spatial_class": "complete HRRR miss",
            "minimum_distance_km": float("nan"),
            "centroid_displacement_km": float("nan"),
            "overlap_fraction": 0.0,
            "coverage_within_6km": 0.0,
            "coverage_within_18km": 0.0,
            "coverage_within_36km": 0.0,
            "displacement_dx_km": float("nan"),
            "displacement_dy_km": float("nan"),
        }
    distance = distance_transform_edt(~hrrr) * 2.0
    minimum = float(np.min(distance[observed]))
    oy, ox = np.argwhere(observed).mean(axis=0)
    hy, hx = np.argwhere(hrrr).mean(axis=0)
    dx, dy = float((hx - ox) * 2.0), float((hy - oy) * 2.0)
    overlap = float(np.sum(observed & hrrr) / np.sum(observed))
    spatial_class = (
        "direct overlap"
        if overlap > 0
        else ("near miss" if minimum <= 36 else "distant false forecast")
    )
    return {
        "hrrr_spatial_class": spatial_class,
        "minimum_distance_km": minimum,
        "centroid_displacement_km": float(np.hypot(dx, dy)),
        "overlap_fraction": overlap,
        "coverage_within_6km": float(np.mean(distance[observed] <= 6)),
        "coverage_within_18km": float(np.mean(distance[observed] <= 18)),
        "coverage_within_36km": float(np.mean(distance[observed] <= 36)),
        "displacement_dx_km": dx,
        "displacement_dy_km": dy,
    }


def _onset_bin(minutes: float) -> str:
    if not np.isfinite(minutes):
        return "unresolved"
    lower = 0 if minutes <= 30 else 30 if minutes <= 60 else 60 if minutes <= 90 else 90
    upper = 30 if lower == 0 else lower + 30
    return f"{lower}-{upper} min"


def run() -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    cache = pd.read_csv("artifacts/stage_6/stage6_cache_manifest.csv")
    frozen = pd.read_csv("artifacts/stage_4c/frozen_holdout/stage4c_final_holdout_manifest.csv")
    table = cache.merge(
        frozen,
        left_on=["event_id", "object_id"],
        right_on=["source_event_id", "object_id"],
        suffixes=("", "_frozen"),
        validate="one_to_one",
    )
    stage6_rows = pd.read_csv("artifacts/stage_6/evaluation/all_initiation_by_row.csv")
    metric = stage6_rows.pivot(
        index="identity", columns="model", values=["f1", "recall", "false_initiation_fraction"]
    )
    positive_rows, hard_rows = [], []
    for row in table.itertuples(index=False):
        source = np.load(row.tensor_path)
        cached = np.load(row.stage6_cache_path)
        observed = np.where(source["target_valid_mask"] > 0, source["target_rate_mm_hr"], np.nan)
        observed_event = np.any(observed >= RAIN, axis=0)
        a_probability = 1 / (1 + np.exp(-cached["aplus_logits"][:, 0]))
        hourly = cached["hrrr_apcp_hourly"]
        h_probability = np.stack(
            [(hourly[0] >= RAIN).astype(float)] * 10 + [(hourly[1] >= RAIN).astype(float)] * 10
        )
        hybrid = 0.5 * a_probability + 0.5 * h_probability
        identity = f"{row.event_id}|{row.object_id}|{row.representative_issue_time_utc}"
        if row.row_role == "clean_initiation":
            onset_map = first_crossing_minutes(observed, RAIN, 6)
            true_onset = float(np.nanmedian(onset_map[observed_event]))
            interval_index = 0 if true_onset <= 60 else 1
            hmask = hourly[interval_index] >= RAIN
            spatial = _spatial(observed_event, hmask)
            af, hf = metric.loc[identity, ("f1", "A_plus")], metric.loc[identity, ("f1", "hybrid")]
            ar, hr = (
                metric.loc[identity, ("recall", "A_plus")],
                metric.loc[identity, ("recall", "hybrid")],
            )
            aff, hff = (
                metric.loc[identity, ("false_initiation_fraction", "A_plus")],
                metric.loc[identity, ("false_initiation_fraction", "hybrid")],
            )
            if hr > ar and hff <= aff:
                help_label = "HRRR adds previously missed initiation"
            elif hf > af:
                help_label = "HRRR helps A+"
            elif hff > aff:
                help_label = "HRRR false addition"
            elif hf < af:
                help_label = "HRRR hurts A+"
            elif ar > 0 and hr > 0:
                help_label = "both correct"
            else:
                help_label = "both miss"
            a_onset = _median_onset(a_probability, POLICY["A_plus"], observed_event)
            h_onset = _median_onset(h_probability, POLICY["raw_HRRR"], observed_event)
            hybrid_onset = _median_onset(hybrid, POLICY["hybrid"], observed_event)
            positive_rows.append(
                {
                    "identity": identity,
                    "event_id": row.event_id,
                    "issue_time_utc": row.representative_issue_time_utc,
                    "object_id": row.object_id,
                    "a_plus_initiation_probability": float(
                        np.mean(np.max(a_probability[:, observed_event], axis=0))
                    ),
                    "hrrr_occurrence_fraction": float(np.mean(hmask[observed_event])),
                    "hybrid_initiation_probability": float(
                        np.mean(np.max(hybrid[:, observed_event], axis=0))
                    ),
                    "observed_onset_minute": true_onset,
                    "a_plus_predicted_onset_minute": a_onset,
                    "hrrr_predicted_onset_minute": h_onset,
                    "hybrid_predicted_onset_minute": hybrid_onset,
                    "hrrr_interval_containing_observed_onset": "0-60"
                    if interval_index == 0
                    else "60-120",
                    "distance_to_nearest_hrrr_interval_boundary_minutes": float(
                        min(abs(true_onset - x) for x in (0, 60, 120))
                    ),
                    "hrrr_correct_interval_at_location": bool(np.any(hmask & observed_event)),
                    "hybrid_timing_error_minutes": hybrid_onset - true_onset,
                    "hybrid_substantially_early": bool(
                        np.isfinite(hybrid_onset) and hybrid_onset < true_onset - 30
                    ),
                    "hybrid_substantially_late": bool(
                        np.isfinite(hybrid_onset) and hybrid_onset > true_onset + 30
                    ),
                    "onset_lead_group": _onset_bin(true_onset),
                    "a_plus_state": _state(a_probability, POLICY["A_plus"], observed_event),
                    "hrrr_state": _state(h_probability, POLICY["raw_HRRR"], observed_event),
                    "hybrid_state": _state(hybrid, POLICY["hybrid"], observed_event),
                    "radar_limited_initiation": bool(row.radar_limited_initiation),
                    "radar_poor_initiation_v2": bool(row.radar_poor_initiation_v2),
                    "diagnostic_label": help_label,
                    "a_plus_f1": float(af),
                    "hybrid_f1": float(hf),
                    "hybrid_minus_a_plus_f1": float(hf - af),
                    "pysteps_coverage": float(row.pysteps_coverage),
                    "final_radar_wet_fraction": float(
                        np.mean(source["radar_rate_mm_hr"][-1] >= RAIN)
                    ),
                    "history_radar_wet_fraction": float(
                        np.mean(source["radar_rate_mm_hr"] >= RAIN)
                    ),
                    "observed_initiation_area_km2": float(np.sum(observed_event) * 4),
                    **spatial,
                }
            )
        elif row.row_role == "hard_negative":
            a_wet = np.any(a_probability >= POLICY["A_plus"], axis=0)
            h_wet = np.any(h_probability >= POLICY["raw_HRRR"], axis=0)
            hybrid_wet = np.any(hybrid >= POLICY["hybrid"], axis=0)
            if h_wet.any() and not a_wet.any():
                alarm = "HRRR-only"
            elif a_wet.any() and not h_wet.any():
                alarm = "A+-only"
            elif a_wet.any() and h_wet.any():
                alarm = "both"
            elif hybrid_wet.any():
                alarm = "threshold interaction"
            else:
                alarm = "none"
            hard_rows.append(
                {
                    "event_id": row.event_id,
                    "issue_time_utc": row.representative_issue_time_utc,
                    "object_id": row.object_id,
                    "hrrr_triggered": bool(h_wet.any()),
                    "hrrr_wet_area_fraction": float(np.mean(h_wet)),
                    "a_plus_mean_probability": float(np.mean(a_probability)),
                    "a_plus_max_probability": float(np.max(a_probability)),
                    "a_plus_wet_area_fraction": float(np.mean(a_wet)),
                    "hybrid_wet_area_fraction": float(np.mean(hybrid_wet)),
                    "blend_crossed_because_of_hrrr": bool(np.any(hybrid_wet & ~a_wet & h_wet)),
                    "false_alarm_source": alarm,
                    "false_area_pattern": "widespread"
                    if np.mean(hybrid_wet) >= 0.01
                    else "isolated",
                }
            )
    positive = pd.DataFrame(positive_rows)
    hard = pd.DataFrame(hard_rows)
    positive.to_csv(ROOT / "row_level_error_table.csv", index=False)
    hard.to_csv(ROOT / "hard_negative_failure_table.csv", index=False)
    lifecycle = stage6_rows.merge(
        positive[["identity", "onset_lead_group", "observed_onset_minute"]], on="identity"
    )
    lifecycle.groupby(["onset_lead_group", "model"], as_index=False).mean(numeric_only=True).to_csv(
        ROOT / "temporal_performance_by_onset_group.csv", index=False
    )
    positive.groupby("onset_lead_group", as_index=False).agg(
        rows=("identity", "size"),
        correct_hrrr_interval_fraction=("hrrr_correct_interval_at_location", "mean"),
        mean_boundary_distance_minutes=(
            "distance_to_nearest_hrrr_interval_boundary_minutes",
            "mean",
        ),
        substantially_early_fraction=("hybrid_substantially_early", "mean"),
        substantially_late_fraction=("hybrid_substantially_late", "mean"),
    ).to_csv(ROOT / "temporal_mismatch_summary.csv", index=False)
    positive.groupby("hrrr_spatial_class", as_index=False).agg(
        rows=("identity", "size"),
        mean_minimum_distance_km=("minimum_distance_km", "mean"),
        mean_centroid_displacement_km=("centroid_displacement_km", "mean"),
        mean_overlap_fraction=("overlap_fraction", "mean"),
        mean_f1_change=("hybrid_minus_a_plus_f1", "mean"),
    ).to_csv(ROOT / "spatial_displacement_summary.csv", index=False)
    v2 = positive[positive.radar_poor_initiation_v2].copy()
    v2["v2_signal_class"] = np.select(
        [
            v2.overlap_fraction > 0,
            v2.minimum_distance_km <= 36,
            v2.hrrr_correct_interval_at_location,
            v2.hrrr_occurrence_fraction <= 0,
        ],
        [
            "HRRR rain directly overlaps eventual initiation",
            "HRRR rain lies nearby but displaced",
            "HRRR predicts correct interval but wrong location",
            "HRRR contains no useful signal",
        ],
        default="HRRR predicts wrong interval",
    )
    v2.to_csv(ROOT / "strict_v2_postmortem.csv", index=False)
    contrast = positive[positive.radar_limited_initiation].copy()
    contrast["cohort"] = np.where(
        contrast.radar_poor_initiation_v2, "strict V2", "RADAR_LIMITED not V2"
    )
    contrast.groupby("cohort", as_index=False).agg(
        rows=("identity", "size"),
        mean_a_plus_probability=("a_plus_initiation_probability", "mean"),
        mean_pysteps_coverage=("pysteps_coverage", "mean"),
        mean_final_radar_wet_fraction=("final_radar_wet_fraction", "mean"),
        mean_hrrr_occurrence=("hrrr_occurrence_fraction", "mean"),
        mean_hrrr_minimum_distance_km=("minimum_distance_km", "mean"),
        mean_onset_lead_minutes=("observed_onset_minute", "mean"),
        mean_hybrid_f1_change=("hybrid_minus_a_plus_f1", "mean"),
    ).to_csv(ROOT / "radar_limited_vs_v2.csv", index=False)
    event = positive.groupby("event_id", as_index=False).agg(
        rows=("identity", "size"),
        hrrr_direct_overlap_fraction=("hrrr_correct_interval_at_location", "mean"),
        mean_hrrr_displacement_km=("minimum_distance_km", "mean"),
        mean_onset_minute=("observed_onset_minute", "mean"),
        mean_a_plus_probability=("a_plus_initiation_probability", "mean"),
        mean_hrrr_occurrence=("hrrr_occurrence_fraction", "mean"),
        mean_observed_area_km2=("observed_initiation_area_km2", "mean"),
        mean_f1_change=("hybrid_minus_a_plus_f1", "mean"),
    )
    event_metrics = stage6_rows.groupby(["event_id", "model"], as_index=False).mean(
        numeric_only=True
    )
    event.merge(
        event_metrics.pivot(index="event_id", columns="model", values="recall"), on="event_id"
    ).to_csv(ROOT / "event_dependence_summary.csv", index=False)
    summary = {
        "positive_rows": len(positive),
        "hard_negative_rows": len(hard),
        "diagnostic_only": True,
        "selection_or_fitting_performed": False,
        "dominant_failure_class_pending_documented_synthesis": True,
        "stage6_conclusion_changed": False,
    }
    (ROOT / "diagnostic_integrity.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
