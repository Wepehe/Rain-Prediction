"""Evaluate the frozen C1 checkpoint on DEV without opening the 2023 holdout."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..evaluation.metrics import brier_score, categorical_metrics, fractions_skill_score_km
from ..models.radar_convlstm import expected_rate_mm_hr
from ..models.radar_thermodynamics import RadarThermodynamicConvLSTM
from .stage4b_holdout import _row_metrics
from .stage4c_train import Dataset

LEADS = (30, 60, 90, 120)
RAIN_THRESHOLDS = (0.1, 1.0, 2.5, 5.0)
FSS_RADII_KM = (6.0, 18.0, 36.0)
P_THRESHOLDS = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2))
VARIANTS = (
    "normal",
    "zeroed",
    "shuffled_global",
    "shuffled_within_event",
    "moisture_zeroed",
    "instability_zeroed",
)


def _bucket() -> dict[str, Any]:
    return {
        "hits": 0,
        "misses": 0,
        "false_alarms": 0,
        "correct_negatives": 0,
        "brier": [],
        **{f"fss_{int(r)}": [] for r in FSS_RADII_KM},
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _finish(bucket: dict[str, Any]) -> dict[str, float | int]:
    hits, misses, false = bucket["hits"], bucket["misses"], bucket["false_alarms"]
    result: dict[str, float | int] = {
        key: bucket[key] for key in ("hits", "misses", "false_alarms", "correct_negatives")
    }
    result.update(
        {
            "csi": _ratio(hits, hits + misses + false),
            "pod": _ratio(hits, hits + misses),
            "far": _ratio(false, hits + false),
            "f1": _ratio(2 * hits, 2 * hits + misses + false),
            "brier_score": float(np.nanmean(bucket["brier"])),
        }
    )
    for radius in FSS_RADII_KM:
        result[f"fss_radius_{int(radius)}km"] = float(np.nanmean(bucket[f"fss_{int(radius)}"]))
    return result


def _condition_inputs(examples, ids, global_perm, within_perm):
    base = torch.stack([examples[index].thermo_inputs for index in ids])
    inputs = {
        "normal": base,
        "zeroed": torch.zeros_like(base),
        "shuffled_global": torch.stack([examples[global_perm[i]].thermo_inputs for i in ids]),
        "shuffled_within_event": torch.stack([examples[within_perm[i]].thermo_inputs for i in ids]),
        "moisture_zeroed": base.clone(),
        "instability_zeroed": base.clone(),
    }
    # T2M, DPT2M, CAPE, CIN, PWAT, then five validity-mask channels.
    inputs["moisture_zeroed"][:, :, [1, 4]] = 0
    inputs["instability_zeroed"][:, :, 2:4] = 0
    return inputs


def run() -> dict[str, object]:
    output = Path("artifacts/stage_4c/c1/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    rule_path = Path("artifacts/stage_4c/c1/dev_selection_rule.json")
    if not rule_path.exists():
        raise FileNotFoundError("DEV selection rule must exist before scoring")

    manifest = pd.read_csv("artifacts/stage_4c/tensors/stage4c_thermodynamic_tensor_manifest.csv")
    table = manifest[manifest.split.isin({"dev", "stage4_dev_repair"})].reset_index(drop=True)
    if table.split.str.contains("holdout", case=False).any() or len(table) != 24:
        raise RuntimeError("Expected exactly 24 non-holdout DEV tensors")
    checkpoint = Path("artifacts/stage_4c/c1/model_best.pt")
    metadata = json.loads(Path("artifacts/stage_4c/c1/training_metadata.json").read_text())
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != metadata["best_checkpoint_sha256"]:
        raise RuntimeError("C1 checkpoint hash differs from training metadata")
    radar_norm = json.loads(
        Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text()
    )["normalization"]["radar"]
    thermo_norm = json.loads(
        Path("artifacts/stage_4c/gate/thermodynamic_normalization_train_only.json").read_text()
    )
    dataset = Dataset(table, {"dev", "stage4_dev_repair"}, radar_norm, thermo_norm)
    examples = [dataset[index] for index in range(len(dataset))]
    events = np.asarray([str(example.metadata["event_id"]) for example in examples])
    global_perm, within_perm = np.roll(np.arange(len(dataset)), 1), np.arange(len(dataset))
    for event_id in np.unique(events):
        indices = np.where(events == event_id)[0]
        if len(indices) > 1:
            within_perm[indices] = np.roll(indices, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarThermodynamicConvLSTM().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    predictions = {name: [] for name in VARIANTS}
    observed, histories = [], []
    for start in range(0, len(dataset), 4):
        ids = np.arange(start, min(start + 4, len(dataset)))
        radar = torch.stack([examples[index].radar_inputs for index in ids]).to(device)
        inputs = {
            name: value.to(device)
            for name, value in _condition_inputs(examples, ids, global_perm, within_perm).items()
        }
        with torch.no_grad():
            for name, value in inputs.items():
                logits, intensity = model(radar, value)
                rates = expected_rate_mm_hr(logits, intensity).cpu().numpy()
                probabilities = torch.sigmoid(logits).cpu().numpy()
                predictions[name].extend(zip(rates, probabilities, strict=True))
        for index in ids:
            example = examples[index]
            observed.append(
                np.where(
                    example.target_mask.numpy() > 0,
                    np.expm1(example.target_intensity.numpy()),
                    np.nan,
                )
            )
            radar_array = example.radar_inputs.numpy()
            recovered = np.expm1(radar_array[:, 0] * radar_norm["std"] + radar_norm["mean"])
            histories.append(np.where(radar_array[:, 1] > 0, recovered, np.nan))
        print(f"[stage4c-eval] {min(start + 4, len(dataset))}/{len(dataset)}", flush=True)

    buckets = defaultdict(_bucket)
    for sample_index in range(len(dataset)):
        for name in VARIANTS:
            rate, probability = predictions[name][sample_index]
            for lead in LEADS:
                lead_index = lead // 6 - 1
                obs, pred = observed[sample_index][lead_index], rate[lead_index]
                for threshold_index, threshold in enumerate(RAIN_THRESHOLDS):
                    bucket = buckets[(name, lead, threshold)]
                    metrics = categorical_metrics(obs, pred, threshold)
                    for key in ("hits", "misses", "false_alarms", "correct_negatives"):
                        bucket[key] += metrics[key]
                    bucket["brier"].append(
                        brier_score(
                            (obs >= threshold).astype(float),
                            probability[lead_index, threshold_index],
                        )
                    )
                    for radius in FSS_RADII_KM:
                        bucket[f"fss_{int(radius)}"].append(
                            fractions_skill_score_km(obs, pred, threshold, radius, 2.0)
                        )
    broad = pd.DataFrame(
        [
            {
                "condition": name,
                "lead_minutes": lead,
                "threshold_mm_hr": threshold,
                **_finish(bucket),
            }
            for (name, lead, threshold), bucket in sorted(buckets.items())
        ]
    )
    broad.to_csv(output / "dev_broad_metrics.csv", index=False)

    lifecycle_rows = []
    positive = np.where(table.event_class.astype(str).eq("positive_initiation"))[0]
    for name in VARIANTS:
        for probability_threshold in P_THRESHOLDS:
            for sample_index in positive:
                _, probability = predictions[name][sample_index]
                dry = np.isfinite(histories[sample_index][-1]) & (histories[sample_index][-1] < 0.1)
                lifecycle_rows.append(
                    {
                        "condition": name,
                        "probability_threshold": probability_threshold,
                        "anchor_id": table.iloc[sample_index].anchor_id,
                        "event_id": table.iloc[sample_index].event_id,
                        **_row_metrics(
                            observed[sample_index][:, dry],
                            probability[:, 0][:, dry],
                            probability_threshold,
                        ),
                    }
                )
    lifecycle = pd.DataFrame(lifecycle_rows)
    lifecycle.to_csv(output / "dev_lifecycle_by_row.csv", index=False)
    event_metrics = lifecycle.groupby(
        ["condition", "probability_threshold", "event_id"], as_index=False
    ).mean(numeric_only=True)
    event_metrics.to_csv(output / "dev_lifecycle_by_event.csv", index=False)
    event_metrics.groupby(["condition", "probability_threshold"], as_index=False).mean(
        numeric_only=True
    ).to_csv(output / "dev_lifecycle_event_macro.csv", index=False)
    lifecycle.groupby(["condition", "probability_threshold"], as_index=False).mean(
        numeric_only=True
    ).to_csv(output / "dev_lifecycle_row_pooled.csv", index=False)

    negative_indices = np.where(table.event_class.astype(str).eq("hard_negative_candidate"))[0]
    negative_rows = []
    for name in VARIANTS:
        for probability_threshold in P_THRESHOLDS:
            for sample_index in negative_indices:
                _, probability = predictions[name][sample_index]
                probability_01 = probability[:, 0]
                valid = np.isfinite(observed[sample_index])
                negative_rows.append(
                    {
                        "condition": name,
                        "probability_threshold": probability_threshold,
                        "anchor_id": table.iloc[sample_index].anchor_id,
                        "event_id": table.iloc[sample_index].event_id,
                        "mean_probability": float(np.nanmean(probability_01[valid])),
                        "max_probability": float(np.nanmax(probability_01[valid])),
                        "wet_area_fraction": float(
                            np.nanmean(probability_01[valid] >= probability_threshold)
                        ),
                        "false_initiation_fraction": float(
                            np.any(probability_01[valid] >= probability_threshold)
                        ),
                        "brier_score": brier_score(
                            (observed[sample_index][valid] >= 0.1).astype(float),
                            probability_01[valid],
                        ),
                    }
                )
    negative = pd.DataFrame(negative_rows)
    negative.to_csv(output / "dev_hard_negative_by_row.csv", index=False)
    negative.groupby(["condition", "probability_threshold", "event_id"], as_index=False).mean(
        numeric_only=True
    ).to_csv(output / "dev_hard_negative_by_event.csv", index=False)

    result = {
        "checkpoint_sha256": checkpoint_sha,
        "selection_rule_sha256": hashlib.sha256(rule_path.read_bytes()).hexdigest(),
        "samples": len(dataset),
        "positive_initiation_samples": len(positive),
        "hard_negative_samples": len(negative_indices),
        "events": int(table.event_id.nunique()),
        "holdout_scored": False,
        "conditions": list(VARIANTS),
        "probability_thresholds": list(P_THRESHOLDS),
        "fss_radii_km": list(FSS_RADII_KM),
    }
    (output / "evaluation_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
