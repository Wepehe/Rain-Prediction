"""Residual example panels for the frozen H1 DEV checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..models.frozen_residual_fusion import FrozenBaselineResidualFusion
from ..models.radar_convlstm import expected_rate_mm_hr
from .stage5_data import Dataset


def run() -> None:
    output = Path("artifacts/stage_5/h1/evaluation/residual_examples")
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset({"dev", "stage4_dev_repair"})
    model = FrozenBaselineResidualFusion()
    model.load_state_dict(torch.load("artifacts/stage_5/h1/model_best.pt"))
    model.eval()
    candidates = []
    with torch.no_grad():
        for index in range(len(dataset)):
            example = dataset[index]
            logits, intensity, _ = model(
                example.base_logits[None],
                example.base_intensity[None],
                example.base_expected[None],
                example.hrrr[None],
                example.goes[None],
            )
            probability = torch.sigmoid(logits)[0, :, 0].numpy()
            base_probability = torch.sigmoid(example.base_logits)[:, 0].numpy()
            rate = expected_rate_mm_hr(logits, intensity)[0].numpy()
            delta = probability - base_probability
            candidates.append((float(np.mean(delta)), index, probability, rate, delta))
    selected = sorted(candidates)[:2] + sorted(candidates)[-2:]
    for rank, (_, index, probability, rate, delta) in enumerate(selected, 1):
        example = dataset[index]
        source = np.load(example.metadata["radar_goes_tensor_path"])
        lead = 9  # 60 minutes
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
        panels = (
            (example.base_expected[lead].numpy(), "A+ expected rate", "viridis", 0, 5),
            (rate[lead], "H1 expected rate", "viridis", 0, 5),
            (source["target_rate_mm_hr"][lead], "Observed rate", "viridis", 0, 5),
            (delta[lead], "H1 - A+ rain probability", "coolwarm", -0.25, 0.25),
        )
        for axis, (values, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
            image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.axis("off")
            fig.colorbar(image, ax=axis, fraction=0.046)
        fig.suptitle(f"{example.metadata['anchor_id']} ({example.metadata['event_class']})")
        fig.tight_layout()
        fig.savefig(output / f"example_{rank}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    run()
