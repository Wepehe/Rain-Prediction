"""Evaluate the identical-capacity A+-only correction control on DEV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..models.frozen_residual_fusion import FrozenBaselineResidualFusion
from .stage4b_holdout import _row_metrics
from .stage5_data import Dataset


def run() -> dict:
    output = Path("artifacts/stage_5/a_plus_only_control/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset({"dev", "stage4_dev_repair"})
    model = FrozenBaselineResidualFusion()
    model.load_state_dict(torch.load("artifacts/stage_5/a_plus_only_control/model_best.pt"))
    model.eval()
    predictions, observations, dry_masks = [], [], []
    with torch.no_grad():
        for index in range(len(dataset)):
            example = dataset[index]
            logits, _, _ = model(
                example.base_logits[None],
                example.base_intensity[None],
                example.base_expected[None],
                torch.zeros_like(example.hrrr[None]),
                torch.zeros_like(example.goes[None]),
            )
            predictions.append(torch.sigmoid(logits)[0, :, 0].numpy())
            source = np.load(example.metadata["radar_goes_tensor_path"])
            observations.append(
                np.where(source["target_valid_mask"] > 0, source["target_rate_mm_hr"], np.nan)
            )
            radar = source["radar_rate_mm_hr"]
            dry_masks.append(np.isfinite(radar[-1]) & (radar[-1] < 0.1))
    rows = []
    for threshold in np.round(np.arange(0.05, 1, 0.05), 2):
        for index in range(len(dataset)):
            row = dataset.table.iloc[index]
            if row.event_class != "positive_initiation":
                continue
            dry = dry_masks[index]
            rows.append(
                {
                    "threshold": threshold,
                    "event_id": row.event_id,
                    **_row_metrics(
                        observations[index][:, dry], predictions[index][:, dry], threshold
                    ),
                }
            )
    event = (
        pd.DataFrame(rows)
        .groupby(["threshold", "event_id"], as_index=False)
        .mean(numeric_only=True)
    )
    macro = event.groupby("threshold", as_index=False).mean(numeric_only=True)
    macro.to_csv(output / "dev_lifecycle_event_macro.csv", index=False)
    best = macro.sort_values("f1", ascending=False).iloc[0]
    result = {
        "best_threshold": float(best.threshold),
        "best_event_macro_f1": float(best.f1),
        "best_dev_loss": json.loads(
            Path("artifacts/stage_5/a_plus_only_control/training_metadata.json").read_text()
        )["best_dev_loss"],
        "identical_trainable_parameters_to_h1": True,
        "protected_2023_holdout_opened": False,
    }
    (output / "evaluation_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
