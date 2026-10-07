"""Pre-training path, gradient, perturbation, and tiny-overfit gate for D1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ..models.radar_dynamics import RadarDynamicsConvLSTM


def run() -> dict[str, object]:
    torch.set_num_threads(12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = pd.read_csv("artifacts/stage_4d/tensors/stage4d_dynamics_tensor_manifest.csv")
    rows = table[table.split.eq("train")].head(2)
    radar_norm = json.loads(
        Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text()
    )["normalization"]["radar"]
    norm = json.loads(
        Path("artifacts/stage_4d/gate/dynamics_normalization_train_only.json").read_text()
    )
    mean, std = np.asarray(norm["mean"]), np.asarray(norm["std"])
    radar, dynamics, target, masks = [], [], [], []
    for row in rows.itertuples():
        rp, dp = np.load(row.radar_tensor_path), np.load(row.dynamics_tensor_path)
        rate = rp["radar_rate_mm_hr"][:, 32:96, 32:96]
        radar_mask = rp["radar_valid_mask"][:, 32:96, 32:96]
        radar_log = (np.log1p(np.nan_to_num(rate)) - radar_norm["mean"]) / radar_norm["std"]
        radar.append(np.stack([radar_log, radar_mask], 1))
        raw = dp["dynamics"][:, :, 32:96, 32:96]
        valid = dp["dynamics_valid_mask"][:, :, 32:96, 32:96]
        dynamics.append(
            np.concatenate([(raw - mean[None, :, None, None]) / std[None, :, None, None], valid], 1)
        )
        target.append(rp["target_rate_mm_hr"][:, 32:96, 32:96])
        masks.append(rp["target_valid_mask"][:, 32:96, 32:96])
    radar_t = torch.tensor(np.stack(radar), dtype=torch.float32, device=device)
    dynamics_t = torch.tensor(np.stack(dynamics), dtype=torch.float32, device=device)
    target_t = torch.tensor(np.stack(target), dtype=torch.float32, device=device)
    mask_t = torch.tensor(np.stack(masks), dtype=torch.float32, device=device)
    model = RadarDynamicsConvLSTM().to(device)
    model.load_radar_weights(
        torch.load("artifacts/stage_4b/a_plus_radar/model_best.pt", map_location=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    thresholds = torch.tensor([0.1, 1, 2.5, 5], device=device)[None, None, :, None, None]
    occurrence = (target_t[:, :, None] >= thresholds).float()
    log_target = torch.log1p(torch.nan_to_num(target_t))
    losses, gradients = [], {}
    for step in range(160):
        optimizer.zero_grad()
        logits, intensity = model(radar_t, dynamics_t)
        loss = (
            F.binary_cross_entropy_with_logits(logits, occurrence, reduction="none")
            * mask_t[:, :, None]
        ).mean()
        loss += (
            0.25
            * (
                F.smooth_l1_loss(F.softplus(intensity), log_target, reduction="none")
                * mask_t
                * (target_t >= 0.1)
            ).mean()
        )
        loss.backward()
        if step == 2:
            groups = {
                "radar": (model.enc1, model.enc2, model.enc3, model.recurrent),
                "dynamics": (
                    model.thermo_enc1,
                    model.thermo_enc2,
                    model.thermo_enc3,
                    model.thermo_recurrent,
                ),
                "fusion": (model.thermo_to_radar,),
            }
            gradients = {
                name: sum(
                    float(p.grad.abs().sum())
                    for module in modules
                    for p in module.parameters()
                    if p.grad is not None
                )
                for name, modules in groups.items()
            }
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        base = model(radar_t, dynamics_t)[0]
        variants = {"all_zero": torch.zeros_like(dynamics_t), "shuffled": dynamics_t.flip(0)}
        for name, channels in (
            ("convergence_zero", [5]),
            ("vertical_motion_zero", [4]),
            ("shear_zero", [7, 8, 9]),
        ):
            value = dynamics_t.clone()
            value[:, :, channels] = 0
            variants[name] = value
        changes = {
            name: float((base - model(radar_t, value)[0]).abs().mean())
            for name, value in variants.items()
        }
    result = {
        "device": str(device),
        "samples": 2,
        "crop": "64x64 pipeline test only",
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / losses[0],
        "gradients": gradients,
        "perturbation_mean_absolute_logit_change": changes,
        "all_paths_nonzero": all(value > 0 for value in gradients.values())
        and all(value > 0 for value in changes.values()),
        "parameter_breakdown": model.parameter_breakdown().__dict__,
        "protected_2023_holdout_used": False,
        "full_training_started": False,
    }
    output = Path("artifacts/stage_4d/tiny_overfit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tiny_overfit.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
