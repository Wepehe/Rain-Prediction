"""DEV-only dependence and initiation evaluation of the Stage 5 non-radar probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..models.frozen_residual_fusion import NonRadarPredictabilityProbe
from .stage4b_holdout import _row_metrics
from .stage5_data import Dataset

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)
CONDITIONS = (
    "normal",
    "hrrr_zeroed",
    "hrrr_shuffled",
    "goes_zeroed",
    "goes_shuffled",
    "all_nonradar_zeroed",
    "all_nonradar_shuffled",
)


def run() -> dict:
    output = Path("artifacts/stage_5/nonradar_probe/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset({"dev", "stage4_dev_repair"})
    examples = [dataset[index] for index in range(len(dataset))]
    order = np.roll(np.arange(len(examples)), 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NonRadarPredictabilityProbe().to(device)
    model.load_state_dict(
        torch.load("artifacts/stage_5/nonradar_probe/model_best.pt", map_location=device)
    )
    model.eval()
    predictions = {condition: [] for condition in CONDITIONS}
    observed, dry_masks = [], []
    with torch.no_grad():
        for index, example in enumerate(examples):
            h, g = example.hrrr[None].to(device), example.goes[None].to(device)
            hs, gs = (
                examples[order[index]].hrrr[None].to(device),
                examples[order[index]].goes[None].to(device),
            )
            inputs = {
                "normal": (h, g),
                "hrrr_zeroed": (torch.zeros_like(h), g),
                "hrrr_shuffled": (hs, g),
                "goes_zeroed": (h, torch.zeros_like(g)),
                "goes_shuffled": (h, gs),
                "all_nonradar_zeroed": (torch.zeros_like(h), torch.zeros_like(g)),
                "all_nonradar_shuffled": (hs, gs),
            }
            for condition, values in inputs.items():
                predictions[condition].append(torch.sigmoid(model(*values))[0].cpu().numpy())
            source = np.load(example.metadata["radar_goes_tensor_path"])
            target = source["target_rate_mm_hr"]
            mask = source["target_valid_mask"] > 0
            observed.append(np.where(mask, target, np.nan))
            radar = source["radar_rate_mm_hr"]
            dry_masks.append(np.isfinite(radar[-1]) & (radar[-1] < 0.1))
            print(f"[stage5-probe-eval] {index + 1}/{len(examples)}", flush=True)
    positive = [
        i for i, x in enumerate(examples) if x.metadata["event_class"] == "positive_initiation"
    ]
    rows = []
    for condition in CONDITIONS:
        for threshold in THRESHOLDS:
            for index in positive:
                dry = dry_masks[index]
                rows.append(
                    {
                        "condition": condition,
                        "probability_threshold": threshold,
                        "anchor_id": examples[index].metadata["anchor_id"],
                        "event_id": examples[index].metadata["event_id"],
                        **_row_metrics(
                            observed[index][:, dry],
                            predictions[condition][index][:, dry],
                            threshold,
                        ),
                    }
                )
    row_table = pd.DataFrame(rows)
    row_table.to_csv(output / "dev_lifecycle_by_row.csv", index=False)
    event = row_table.groupby(
        ["condition", "probability_threshold", "event_id"], as_index=False
    ).mean(numeric_only=True)
    event.to_csv(output / "dev_lifecycle_by_event.csv", index=False)
    macro = event.groupby(["condition", "probability_threshold"], as_index=False).mean(
        numeric_only=True
    )
    macro.to_csv(output / "dev_lifecycle_event_macro.csv", index=False)
    negatives = [
        i for i, x in enumerate(examples) if x.metadata["event_class"] == "hard_negative_candidate"
    ]
    hard_rows = []
    for condition in CONDITIONS:
        for threshold in THRESHOLDS:
            for index in negatives:
                probability = predictions[condition][index]
                valid = np.isfinite(observed[index])
                hard_rows.append(
                    {
                        "condition": condition,
                        "probability_threshold": threshold,
                        "anchor_id": examples[index].metadata["anchor_id"],
                        "event_id": examples[index].metadata["event_id"],
                        "mean_probability": float(np.mean(probability[valid])),
                        "max_probability": float(np.max(probability[valid])),
                        "wet_area_fraction": float(np.mean(probability[valid] >= threshold)),
                        "false_initiation_fraction": float(np.any(probability[valid] >= threshold)),
                        "brier_score": float(
                            np.mean((probability[valid] - (observed[index][valid] >= 0.1)) ** 2)
                        ),
                    }
                )
    hard = pd.DataFrame(hard_rows)
    hard.to_csv(output / "dev_hard_negative_by_row.csv", index=False)
    hard_event = hard.groupby(
        ["condition", "probability_threshold", "event_id"], as_index=False
    ).mean(numeric_only=True)
    hard_event.to_csv(output / "dev_hard_negative_by_event.csv", index=False)
    normal = macro[macro.condition == "normal"].sort_values("f1", ascending=False).iloc[0]
    selected = float(normal.probability_threshold)
    selected_rows = macro[macro.probability_threshold == selected].set_index("condition")
    result = {
        "samples": len(examples),
        "positive_initiation_rows": len(positive),
        "positive_initiation_events": len({examples[i].metadata["event_id"] for i in positive}),
        "hard_negative_rows": len(negatives),
        "selected_dev_threshold": selected,
        "normal_event_macro_f1": float(selected_rows.loc["normal", "f1"]),
        "f1_by_condition_at_selected_threshold": selected_rows.f1.to_dict(),
        "diagnostic_only": True,
        "radar_input_used": False,
        "protected_2023_holdout_opened": False,
    }
    (output / "evaluation_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
