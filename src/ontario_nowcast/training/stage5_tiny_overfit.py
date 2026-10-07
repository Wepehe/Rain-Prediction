"""Tiny optimization and frozen-baseline integrity gate for H1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ..models.frozen_residual_fusion import FrozenBaselineResidualFusion
from .stage5_data import Dataset, collate


def run() -> dict[str, object]:
    torch.set_num_threads(12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset({"train"}, slice(32, 96))
    batch = collate([dataset[0], dataset[1]])
    batch = {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }
    model = FrozenBaselineResidualFusion().to(device)
    checkpoint = Path("artifacts/stage_4b/a_plus_radar/model_best.pt")
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with torch.no_grad():
        initial = model(
            batch["base_logits"],
            batch["base_intensity"],
            batch["base_expected"],
            batch["hrrr"],
            batch["goes"],
        )
    initial_exact = torch.equal(initial[0], batch["base_logits"]) and torch.equal(
        initial[1], batch["base_intensity"]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    gradients = {}
    for step in range(160):
        optimizer.zero_grad()
        hrrr = batch["hrrr"].detach().clone().requires_grad_(True)
        logits, intensity, _raw = model(
            batch["base_logits"],
            batch["base_intensity"],
            batch["base_expected"],
            hrrr,
            batch["goes"],
        )
        loss = (
            F.binary_cross_entropy_with_logits(logits, batch["target_occurrence"], reduction="none")
            * batch["target_mask"][:, :, None]
        ).mean()
        loss += (
            0.25
            * (
                F.smooth_l1_loss(F.softplus(intensity), batch["target_intensity"], reduction="none")
                * batch["target_mask"]
                * batch["target_occurrence"][:, :, 0]
            ).mean()
        )
        loss.backward()
        if step == 2:
            gradients = {
                "hrrr_input": float(hrrr.grad.abs().sum()),
                "shared_aplus_hrrr_context": sum(
                    float(p.grad.abs().sum()) for p in model.context.parameters()
                ),
                "goes_encoder": sum(
                    float(p.grad.abs().sum())
                    for module in (model.goes1, model.goes2, model.goes3, model.goes_recurrent)
                    for p in module.parameters()
                ),
                "fusion": sum(float(p.grad.abs().sum()) for p in model.fusion[0].parameters()),
                "residual_logit_output": float(model.fusion[-1].weight.grad[:4].abs().sum()),
                "residual_intensity_output": float(model.fusion[-1].weight.grad[4:].abs().sum()),
            }
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    variants = {
        "hrrr_zero": (torch.zeros_like(batch["hrrr"]), batch["goes"]),
        "hrrr_shuffle": (batch["hrrr"].flip(0), batch["goes"]),
        "goes_zero": (batch["hrrr"], torch.zeros_like(batch["goes"])),
        "goes_shuffle": (batch["hrrr"], batch["goes"].flip(0)),
        "all_zero": (torch.zeros_like(batch["hrrr"]), torch.zeros_like(batch["goes"])),
        "all_shuffle": (batch["hrrr"].flip(0), batch["goes"].flip(0)),
    }
    with torch.no_grad():
        normal = model(
            batch["base_logits"],
            batch["base_intensity"],
            batch["base_expected"],
            batch["hrrr"],
            batch["goes"],
        )
        sensitivity = {
            name: float(
                (
                    normal[0]
                    - model(
                        batch["base_logits"], batch["base_intensity"], batch["base_expected"], h, g
                    )[0]
                )
                .abs()
                .mean()
            )
            for name, (h, g) in variants.items()
        }
        delta_logit = 4 * torch.tanh(normal[2][:, :, :4])
        delta_intensity = 2 * torch.tanh(normal[2][:, :, 4])
    after_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / losses[0],
        "initial_exact_a_plus": initial_exact,
        "initial_residual_exactly_zero": bool(torch.count_nonzero(initial[2]) == 0),
        "gradients": gradients,
        "all_required_gradients_nonzero": all(value > 0 for value in gradients.values()),
        "post_optimization_residual_nonzero": bool(torch.count_nonzero(normal[2]) > 0),
        "sensitivity_mean_absolute_logit_change": sensitivity,
        "delta_logit_min_max": [float(delta_logit.min()), float(delta_logit.max())],
        "delta_intensity_min_max": [float(delta_intensity.min()), float(delta_intensity.max())],
        "finite": bool(torch.isfinite(delta_logit).all() and torch.isfinite(delta_intensity).all()),
        "delta_logit_saturated_fraction": float((delta_logit.abs() > 3.99).float().mean()),
        "delta_intensity_saturated_fraction": float((delta_intensity.abs() > 1.99).float().mean()),
        "aplus_checkpoint_bit_identical": before_hash == after_hash,
        "protected_holdout_used": False,
    }
    output = Path("artifacts/stage_5/tiny_overfit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tiny_overfit.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
