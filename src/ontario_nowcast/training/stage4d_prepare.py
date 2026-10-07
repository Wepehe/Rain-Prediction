"""Validate Stage 4D tensors, compute TRAIN-only normalization, and plot fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stage4d_tensors import VARIABLES


def run(output_dir: Path = Path("artifacts/stage_4d/gate")) -> dict[str, object]:
    manifest_path = Path("artifacts/stage_4d/tensors/stage4d_dynamics_tensor_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    train = manifest[manifest.split.eq("train")]
    count = np.zeros(len(VARIABLES), dtype=np.int64)
    total = np.zeros(len(VARIABLES))
    total_sq = np.zeros(len(VARIABLES))
    raw_min = np.full(len(VARIABLES), np.inf)
    raw_max = np.full(len(VARIABLES), -np.inf)
    for row in train.itertuples(index=False):
        payload = np.load(row.dynamics_tensor_path)
        values = payload["dynamics"].astype(np.float64)
        mask = payload["dynamics_valid_mask"] > 0
        for channel in range(len(VARIABLES)):
            selected = values[:, channel][mask[:, channel]]
            count[channel] += selected.size
            total[channel] += selected.sum()
            total_sq[channel] += np.square(selected).sum()
            raw_min[channel] = min(raw_min[channel], float(selected.min()))
            raw_max[channel] = max(raw_max[channel], float(selected.max()))
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0))
    normalization = {
        "source_split": "train",
        "train_tensors": len(train),
        "variables": list(VARIABLES),
        "transforms": ["identity"] * len(VARIABLES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count.tolist(),
        "raw_min": raw_min.tolist(),
        "raw_max": raw_max.tolist(),
        "clipping": None,
        "units": [
            "m s-1",
            "m s-1",
            "m s-1",
            "m s-1",
            "Pa s-1",
            "s-1",
            "s-1",
            "m s-1",
            "m s-1",
            "m s-1",
        ],
        "tensor_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_path = output_dir / "dynamics_normalization_train_only.json"
    norm_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")

    requested = {
        "convective_initiation": "stage4_dev_initiation_2026_06_18",
        "organized_convection": "june_frontal_convection_2024_06_05",
        "stratiform_precipitation": "september_stratiform_2024_09_24",
        "dissipation": "stage4_dev_dissipation_2026_05_20",
        "dynamically_active_dry": "stage4_dev_dry_favourable_2026_07_10",
    }
    visual_dir = output_dir / "visual_panels"
    visual_dir.mkdir(parents=True, exist_ok=True)
    visual_rows = []
    channel_indices = (0, 1, 4, 5, 6, 9)
    for label, event_id in requested.items():
        row = manifest[manifest.event_id.eq(event_id)].iloc[0]
        tensor = np.load(row.dynamics_tensor_path)["dynamics"][0]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        for axis, channel in zip(axes.ravel(), channel_indices, strict=True):
            image = axis.imshow(tensor[channel], origin="lower", cmap="coolwarm")
            axis.set_title(VARIABLES[channel])
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, shrink=0.75)
        path = visual_dir / f"{label}.png"
        fig.suptitle(f"{label}: {event_id}")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        visual_rows.append(
            {
                "category": label,
                "event_id": event_id,
                "path": path.as_posix(),
                "finite": bool(np.isfinite(tensor).all()),
                "nonconstant_channels": int(sum(np.nanstd(field) > 0 for field in tensor)),
            }
        )
    pd.DataFrame(visual_rows).to_csv(output_dir / "visual_panel_audit.csv", index=False)
    plausible = bool(
        raw_min[9] >= 0
        and raw_max[9] < 150
        and max(abs(raw_min[5]), abs(raw_max[5])) < 0.02
        and max(abs(raw_min[6]), abs(raw_max[6])) < 0.02
        and np.all(std > 0)
    )
    summary = {
        "tensor_rows": len(manifest),
        "train_rows_used_for_normalization": len(train),
        "normalization_source_split": "train",
        "all_std_positive": bool(np.all(std > 0)),
        "physical_range_gate": plausible,
        "visual_panels": len(visual_rows),
        "all_visual_fields_finite": bool(all(row["finite"] for row in visual_rows)),
        "all_visual_channels_nonconstant": bool(
            all(row["nonconstant_channels"] == len(VARIABLES) for row in visual_rows)
        ),
        "protected_2023_holdout_used": False,
        "training_started": False,
    }
    (output_dir / "stage4d_spatial_normalization_gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
