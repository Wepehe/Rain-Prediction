"""One-time protected Stage 6 transparent-hybrid scoring."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..evaluation.metrics import brier_score, categorical_metrics, fractions_skill_score_km
from ..models.baselines import pysteps_deterministic_extrapolation
from .stage4b_holdout import _row_metrics
from .stage4c_eval import _bucket, _finish
from .stage6_prepare import EXPECTED_HOLDOUT

ROOT = Path("artifacts/stage_6")
MODELS = ("A_plus", "raw_HRRR", "hybrid", "PySTEPS")
POLICIES = {"A_plus": 0.35, "raw_HRRR": 0.5, "hybrid": 0.3, "PySTEPS": 0.5}
RAIN_THRESHOLDS = (0.1, 1.0, 2.5, 5.0)
LEADS = (30, 60, 90, 120)


def _bootstrap(values: np.ndarray, seed: int = 62023) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(10000)])
    return tuple(np.quantile(means, [0.025, 0.975]))


def _summaries(rows: pd.DataFrame, cohort: str, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = rows[rows.cohort == cohort]
    row_macro = subset.groupby("model", as_index=False).mean(numeric_only=True)
    event = subset.groupby(["model", "event_id"], as_index=False).mean(numeric_only=True)
    event_macro = event.groupby("model", as_index=False).mean(numeric_only=True)
    cis = []
    for model in MODELS:
        values = event[event.model == model].f1.to_numpy()
        low, high = _bootstrap(values)
        cis.append({"model": model, "f1_event_bootstrap_low": low, "f1_event_bootstrap_high": high})
    event_macro = event_macro.merge(pd.DataFrame(cis), on="model", how="left")
    subset.to_csv(output / f"{cohort}_by_row.csv", index=False)
    event.to_csv(output / f"{cohort}_by_event.csv", index=False)
    row_macro.to_csv(output / f"{cohort}_row_macro.csv", index=False)
    event_macro.to_csv(output / f"{cohort}_event_macro.csv", index=False)
    return row_macro, event_macro


def run() -> dict:
    output = ROOT / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "final_decision.json").exists():
        raise RuntimeError("Stage 6 protected score already exists; refusing a second score")
    integrity = json.loads((ROOT / "materialization_integrity.json").read_text())
    if not integrity["scoreable"] or integrity["holdout_manifest_sha256"] != EXPECTED_HOLDOUT:
        raise RuntimeError("materialization integrity gate failed")
    manifest = pd.read_csv("artifacts/stage_4c/frozen_holdout/stage4c_final_holdout_manifest.csv")
    cache = pd.read_csv(ROOT / "stage6_cache_manifest.csv")
    table = cache.merge(
        manifest,
        left_on=["event_id", "object_id"],
        right_on=["source_event_id", "object_id"],
        suffixes=("", "_frozen"),
        validate="one_to_one",
    )
    predictions, lifecycle_rows, hard_rows = {}, [], []
    broad_buckets, heavy_buckets = defaultdict(_bucket), defaultdict(_bucket)
    for number, row in enumerate(table.itertuples(index=False), 1):
        source = np.load(row.tensor_path)
        cached = np.load(row.stage6_cache_path)
        observed = np.where(source["target_valid_mask"] > 0, source["target_rate_mm_hr"], np.nan)
        radar = source["radar_rate_mm_hr"]
        dry = np.isfinite(radar[-1]) & (radar[-1] < 0.1)
        a_probability = 1 / (1 + np.exp(-cached["aplus_logits"]))
        a_rate = cached["aplus_expected_rate_mm_hr"]
        hourly = cached["hrrr_apcp_hourly"]
        h_rate = np.stack([hourly[0]] * 10 + [hourly[1]] * 10)
        h_probability = (
            h_rate[:, None] >= np.asarray(RAIN_THRESHOLDS)[None, :, None, None]
        ).astype(np.float32)
        hybrid_probability = 0.5 * a_probability + 0.5 * h_probability
        hybrid_rate = 0.5 * a_rate + 0.5 * h_rate
        py_rate = pysteps_deterministic_extrapolation(radar[-3:], 20)
        py_probability = (
            py_rate[:, None] >= np.asarray(RAIN_THRESHOLDS)[None, :, None, None]
        ).astype(np.float32)
        probabilities = {
            "A_plus": a_probability,
            "raw_HRRR": h_probability,
            "hybrid": hybrid_probability,
            "PySTEPS": py_probability,
        }
        rates = {"A_plus": a_rate, "raw_HRRR": h_rate, "hybrid": hybrid_rate, "PySTEPS": py_rate}
        identity = f"{row.event_id}|{row.object_id}|{row.representative_issue_time_utc}"
        predictions[identity] = {
            "observed": observed,
            "dry": dry,
            "probabilities": probabilities,
            "rates": rates,
            "row": row,
        }
        if row.row_role == "clean_initiation":
            cohorts = ["all_initiation"]
            if bool(row.radar_limited_initiation):
                cohorts.append("radar_limited_initiation")
            if bool(row.radar_poor_initiation_v2):
                cohorts.append("radar_poor_initiation_v2")
            for cohort in cohorts:
                for model in MODELS:
                    lifecycle_rows.append(
                        {
                            "cohort": cohort,
                            "identity": identity,
                            "event_id": row.event_id,
                            "object_id": row.object_id,
                            "model": model,
                            "threshold": POLICIES[model],
                            **_row_metrics(
                                observed[:, dry],
                                probabilities[model][:, 0][:, dry],
                                POLICIES[model],
                            ),
                        }
                    )
        if row.row_role == "hard_negative":
            valid = np.isfinite(observed)
            for model in ("A_plus", "raw_HRRR", "hybrid"):
                probability = probabilities[model][:, 0]
                hard_rows.append(
                    {
                        "identity": identity,
                        "event_id": row.event_id,
                        "model": model,
                        "threshold": POLICIES[model],
                        "mean_probability": float(np.mean(probability[valid])),
                        "max_probability": float(np.max(probability[valid])),
                        "wet_area_fraction": float(np.mean(probability[valid] >= POLICIES[model])),
                        "false_initiation_fraction": float(
                            np.any(probability[valid] >= POLICIES[model])
                        ),
                        "brier_score": brier_score(
                            (observed[valid] >= 0.1).astype(float), probability[valid]
                        ),
                    }
                )
        for model in MODELS:
            for lead in LEADS:
                li = lead // 6 - 1
                for ti, threshold in enumerate(RAIN_THRESHOLDS):
                    for buckets in (
                        [broad_buckets]
                        if row.row_role != "heavy_rain_generalization"
                        else [broad_buckets, heavy_buckets]
                    ):
                        bucket = buckets[(model, lead, threshold)]
                        metric = categorical_metrics(observed[li], rates[model][li], threshold)
                        for key in ("hits", "misses", "false_alarms", "correct_negatives"):
                            bucket[key] += metric[key]
                        bucket["brier"].append(
                            brier_score(
                                (observed[li] >= threshold).astype(float),
                                probabilities[model][li, ti],
                            )
                        )
                        for radius in (6.0, 18.0, 36.0):
                            bucket[f"fss_{int(radius)}"].append(
                                fractions_skill_score_km(
                                    observed[li], rates[model][li], threshold, radius, 2.0
                                )
                            )
        print(f"[stage6-score] {number}/{len(table)}", flush=True)
    lifecycle = pd.DataFrame(lifecycle_rows)
    lifecycle.to_csv(output / "lifecycle_all_rows.csv", index=False)
    cohort_results = {}
    for cohort in ("all_initiation", "radar_limited_initiation", "radar_poor_initiation_v2"):
        _row_macro, event_macro = _summaries(lifecycle, cohort, output)
        cohort_results[cohort] = {
            row.model: float(row.f1) for row in event_macro.itertuples(index=False)
        }
    hard = pd.DataFrame(hard_rows)
    hard.to_csv(output / "hard_negative_by_row.csv", index=False)
    hard_event = hard.groupby(["model", "event_id"], as_index=False).mean(numeric_only=True)
    hard_event.to_csv(output / "hard_negative_by_event.csv", index=False)
    hard_macro = hard_event.groupby("model", as_index=False).mean(numeric_only=True)
    hard_macro.to_csv(output / "hard_negative_event_macro.csv", index=False)
    for name, buckets in (("broad_metrics", broad_buckets), ("heavy_rain_metrics", heavy_buckets)):
        pd.DataFrame(
            [
                {
                    "model": model,
                    "lead_minutes": lead,
                    "threshold_mm_hr": threshold,
                    **_finish(bucket),
                }
                for (model, lead, threshold), bucket in sorted(buckets.items())
            ]
        ).to_csv(output / f"{name}.csv", index=False)
    all_event = pd.read_csv(output / "all_initiation_by_event.csv")
    pivot = all_event.pivot(index="event_id", columns="model", values="f1")
    event_gains = (pivot.hybrid - pivot.A_plus).to_dict()
    majority = sum(value > 0 for value in event_gains.values()) >= 2
    all_gain = (
        cohort_results["all_initiation"]["hybrid"] - cohort_results["all_initiation"]["A_plus"]
    )
    limited_gain = (
        cohort_results["radar_limited_initiation"]["hybrid"]
        - cohort_results["radar_limited_initiation"]["A_plus"]
    )
    v2_gain = (
        cohort_results["radar_poor_initiation_v2"]["hybrid"]
        - cohort_results["radar_poor_initiation_v2"]["A_plus"]
    )
    hybrid_hard = hard_macro[hard_macro.model == "hybrid"].iloc[0]
    hard_acceptable = bool(
        hybrid_hard.wet_area_fraction <= 0.01 and hybrid_hard.false_initiation_fraction <= 0.40
    )
    if all_gain > 0 and majority and (limited_gain > 0 or v2_gain > 0) and hard_acceptable:
        classification = "ROBUST HYBRID GAIN"
    elif all_gain > 0:
        classification = "MIXED GENERALIZATION"
    elif abs(all_gain) < 0.01 and limited_gain <= 0 and v2_gain <= 0:
        classification = "NULL"
    else:
        classification = "NEGATIVE"
    # Post-score explanatory grouping; never used for model or threshold selection.
    rows = lifecycle[lifecycle.cohort == "all_initiation"].pivot(
        index=["identity", "event_id", "object_id"], columns="model"
    )
    analyses = []
    for key, values in rows.iterrows():
        af, hf = values[("f1", "A_plus")], values[("f1", "hybrid")]
        ar, hr = values[("recall", "A_plus")], values[("recall", "hybrid")]
        aff, hff = (
            values[("false_initiation_fraction", "A_plus")],
            values[("false_initiation_fraction", "hybrid")],
        )
        if hr > ar and hff <= aff:
            category = "HRRR correctly adds rain missed by A+"
        elif hff < aff and hr >= ar:
            category = "HRRR correctly suppresses/confidence-adjusts A+"
        elif hff > aff:
            category = "HRRR causes false addition"
        elif hf < af:
            category = "A+ was already correct and HRRR hurts"
        else:
            category = "both systems fail"
        analyses.append(
            {
                "identity": key[0],
                "event_id": key[1],
                "object_id": key[2],
                "category": category,
                "hybrid_minus_a_plus_f1": hf - af,
                "hybrid_minus_a_plus_recall": hr - ar,
            }
        )
    analysis = pd.DataFrame(analyses)
    analysis.to_csv(output / "where_hybrid_helps.csv", index=False)
    panel_dir = output / "representative_panels"
    panel_dir.mkdir(exist_ok=True)
    for number, record in enumerate(
        analysis.sort_values("hybrid_minus_a_plus_f1")
        .groupby("category")
        .head(1)
        .itertuples(index=False),
        1,
    ):
        item = predictions[record.identity]
        lead = 9
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        panels = (
            (item["probabilities"]["A_plus"][lead, 0], "A+ probability", "viridis", 0, 1),
            (item["probabilities"]["raw_HRRR"][lead, 0], "HRRR occurrence", "viridis", 0, 1),
            (item["probabilities"]["hybrid"][lead, 0], "Hybrid probability", "viridis", 0, 1),
            (item["observed"][lead], "Observed rate", "magma", 0, 5),
        )
        for axis, (values, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
            image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.axis("off")
            fig.colorbar(image, ax=axis, fraction=0.046)
        fig.suptitle(f"{record.category}: {record.object_id}, +60 min")
        fig.tight_layout()
        fig.savefig(panel_dir / f"panel_{number}.png", dpi=160)
        plt.close(fig)
    result = {
        "classification": classification,
        "all_initiation_event_macro_f1": cohort_results["all_initiation"],
        "radar_limited_event_macro_f1": cohort_results["radar_limited_initiation"],
        "radar_poor_v2_event_macro_f1": cohort_results["radar_poor_initiation_v2"],
        "hybrid_minus_a_plus_all_f1": all_gain,
        "hybrid_minus_a_plus_radar_limited_f1": limited_gain,
        "hybrid_minus_a_plus_v2_f1": v2_gain,
        "event_f1_gains": event_gains,
        "majority_events_improved": majority,
        "hard_negative_acceptable": hard_acceptable,
        "procedure_sha256": integrity["procedure_sha256"],
        "protected_rows_scored_once": 68,
        "h1_scored": False,
        "post_score_tuning": False,
    }
    (output / "final_decision.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
