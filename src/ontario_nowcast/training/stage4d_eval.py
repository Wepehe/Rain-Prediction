"""Frozen DEV-only D1 evaluation and dynamics-dependence pass."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..evaluation.metrics import brier_score, categorical_metrics, fractions_skill_score_km
from ..models.radar_convlstm import expected_rate_mm_hr
from ..models.radar_dynamics import RadarDynamicsConvLSTM
from .stage4b_holdout import _row_metrics
from .stage4c_eval import _bucket, _finish
from .stage4d_train import Dataset

LEADS = (30, 60, 90, 120)
RAIN_THRESHOLDS = (0.1, 1.0, 2.5, 5.0)
P_THRESHOLDS = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2))
CONDITIONS = (
    "normal",
    "zeroed",
    "shuffled_global",
    "shuffled_within_event",
    "convergence_removed",
    "vertical_motion_removed",
    "shear_removed",
)


def _inputs(examples, indices, global_order, within_order):
    base = torch.stack([examples[index].dynamics_inputs for index in indices])
    result = {
        "normal": base,
        "zeroed": torch.zeros_like(base),
        "shuffled_global": torch.stack(
            [examples[global_order[index]].dynamics_inputs for index in indices]
        ),
        "shuffled_within_event": torch.stack(
            [examples[within_order[index]].dynamics_inputs for index in indices]
        ),
        "convergence_removed": base.clone(),
        "vertical_motion_removed": base.clone(),
        "shear_removed": base.clone(),
    }
    result["convergence_removed"][:, :, 5] = 0
    result["vertical_motion_removed"][:, :, 4] = 0
    result["shear_removed"][:, :, 7:10] = 0
    return result


def run() -> dict[str, object]:
    output = Path("artifacts/stage_4d/d1/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path("artifacts/stage_4d/d1/model_best.pt")
    metadata = json.loads(Path("artifacts/stage_4d/d1/training_metadata.json").read_text())
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != metadata["best_checkpoint_sha256"]:
        raise RuntimeError("D1 checkpoint hash mismatch")
    table = pd.read_csv("artifacts/stage_4d/tensors/stage4d_dynamics_tensor_manifest.csv")
    table = table[table.split.isin({"dev", "stage4_dev_repair"})].reset_index(drop=True)
    if len(table) != 24 or table.split.str.contains("holdout", case=False).any():
        raise RuntimeError("D1 evaluation requires exactly 24 non-holdout DEV tensors")
    radar_norm = json.loads(
        Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text()
    )["normalization"]["radar"]
    dynamics_norm = json.loads(
        Path("artifacts/stage_4d/gate/dynamics_normalization_train_only.json").read_text()
    )
    dataset = Dataset(table, {"dev", "stage4_dev_repair"}, radar_norm, dynamics_norm)
    examples = [dataset[index] for index in range(len(dataset))]
    events = np.asarray([str(example.metadata["event_id"]) for example in examples])
    global_order = np.roll(np.arange(len(dataset)), 1)
    within_order = np.arange(len(dataset))
    for event_id in np.unique(events):
        indices = np.where(events == event_id)[0]
        if len(indices) > 1:
            within_order[indices] = np.roll(indices, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarDynamicsConvLSTM().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    predictions = {condition: [] for condition in CONDITIONS}
    observed, histories = [], []
    for start in range(0, len(dataset), 4):
        indices = np.arange(start, min(start + 4, len(dataset)))
        radar = torch.stack([examples[index].radar_inputs for index in indices]).to(device)
        condition_inputs = {
            name: value.to(device)
            for name, value in _inputs(examples, indices, global_order, within_order).items()
        }
        with torch.no_grad():
            for name, value in condition_inputs.items():
                logits, intensity = model(radar, value)
                predictions[name].extend(
                    zip(
                        expected_rate_mm_hr(logits, intensity).cpu().numpy(),
                        torch.sigmoid(logits).cpu().numpy(),
                        strict=True,
                    )
                )
        for index in indices:
            example = examples[index]
            observed.append(
                np.where(
                    example.target_mask.numpy() > 0,
                    np.expm1(example.target_intensity.numpy()),
                    np.nan,
                )
            )
            radar_array = example.radar_inputs.numpy()
            rate = np.expm1(radar_array[:, 0] * radar_norm["std"] + radar_norm["mean"])
            histories.append(np.where(radar_array[:, 1] > 0, rate, np.nan))
        print(f"[stage4d-eval] {min(start + 4, len(dataset))}/{len(dataset)}", flush=True)
    buckets = defaultdict(_bucket)
    for sample_index in range(len(dataset)):
        for condition in CONDITIONS:
            rate, probability = predictions[condition][sample_index]
            for lead in LEADS:
                lead_index = lead // 6 - 1
                obs, pred = observed[sample_index][lead_index], rate[lead_index]
                for threshold_index, threshold in enumerate(RAIN_THRESHOLDS):
                    bucket = buckets[(condition, lead, threshold)]
                    metric = categorical_metrics(obs, pred, threshold)
                    for key in ("hits", "misses", "false_alarms", "correct_negatives"):
                        bucket[key] += metric[key]
                    bucket["brier"].append(
                        brier_score(
                            (obs >= threshold).astype(float),
                            probability[lead_index, threshold_index],
                        )
                    )
                    for radius in (6.0, 18.0, 36.0):
                        bucket[f"fss_{int(radius)}"].append(
                            fractions_skill_score_km(obs, pred, threshold, radius, 2.0)
                        )
    pd.DataFrame(
        [
            {
                "condition": condition,
                "lead_minutes": lead,
                "threshold_mm_hr": threshold,
                **_finish(bucket),
            }
            for (condition, lead, threshold), bucket in sorted(buckets.items())
        ]
    ).to_csv(output / "dev_broad_metrics.csv", index=False)
    lifecycle_rows = []
    positive = np.where(table.event_class.eq("positive_initiation"))[0]
    for condition in CONDITIONS:
        for threshold in P_THRESHOLDS:
            for sample_index in positive:
                _, probability = predictions[condition][sample_index]
                dry = np.isfinite(histories[sample_index][-1]) & (histories[sample_index][-1] < 0.1)
                lifecycle_rows.append(
                    {
                        "condition": condition,
                        "probability_threshold": threshold,
                        "anchor_id": table.iloc[sample_index].anchor_id,
                        "event_id": table.iloc[sample_index].event_id,
                        **_row_metrics(
                            observed[sample_index][:, dry], probability[:, 0][:, dry], threshold
                        ),
                    }
                )
    lifecycle = pd.DataFrame(lifecycle_rows)
    lifecycle.to_csv(output / "dev_lifecycle_by_row.csv", index=False)
    event = lifecycle.groupby(
        ["condition", "probability_threshold", "event_id"], as_index=False
    ).mean(numeric_only=True)
    event.to_csv(output / "dev_lifecycle_by_event.csv", index=False)
    event.groupby(["condition", "probability_threshold"], as_index=False).mean(
        numeric_only=True
    ).to_csv(output / "dev_lifecycle_event_macro.csv", index=False)
    negatives = np.where(table.event_class.eq("hard_negative_candidate"))[0]
    negative_rows = []
    for condition in CONDITIONS:
        for threshold in P_THRESHOLDS:
            for sample_index in negatives:
                probability = predictions[condition][sample_index][1][:, 0]
                valid = np.isfinite(observed[sample_index])
                negative_rows.append(
                    {
                        "condition": condition,
                        "probability_threshold": threshold,
                        "anchor_id": table.iloc[sample_index].anchor_id,
                        "event_id": table.iloc[sample_index].event_id,
                        "mean_probability": float(np.nanmean(probability[valid])),
                        "max_probability": float(np.nanmax(probability[valid])),
                        "wet_area_fraction": float(np.mean(probability[valid] >= threshold)),
                        "false_initiation_fraction": float(np.any(probability[valid] >= threshold)),
                        "brier_score": brier_score(
                            (observed[sample_index][valid] >= 0.1).astype(float), probability[valid]
                        ),
                    }
                )
    negative = pd.DataFrame(negative_rows)
    negative.to_csv(output / "dev_hard_negative_by_row.csv", index=False)
    negative.groupby(["condition", "probability_threshold", "event_id"], as_index=False).mean(
        numeric_only=True
    ).to_csv(output / "dev_hard_negative_by_event.csv", index=False)
    result = {
        "checkpoint_sha256": checkpoint_hash,
        "samples": len(dataset),
        "positive_initiation_samples": len(positive),
        "hard_negative_samples": len(negatives),
        "conditions": list(CONDITIONS),
        "holdouts_scored": False,
    }
    (output / "evaluation_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
