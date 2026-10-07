"""Pre-training gradient, perturbation, and tiny-overfit gate for C1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ..models.radar_thermodynamics import RadarThermodynamicConvLSTM
from .stage4c_prepare import _transform


def run() -> dict[str, object]:
    torch.set_num_threads(12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tm = pd.read_csv("artifacts/stage_4c/tensors/stage4c_thermodynamic_tensor_manifest.csv")
    rows = tm[tm.split.eq("train")].head(2)
    rn = json.loads(Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text())[
        "normalization"
    ]["radar"]
    tn = json.loads(
        Path("artifacts/stage_4c/gate/thermodynamic_normalization_train_only.json").read_text()
    )
    mean = np.asarray(tn["mean"])
    std = np.asarray(tn["std"])
    radar = []
    thermo = []
    target = []
    mask = []
    for row in rows.itertuples():
        rp = np.load(row.radar_tensor_path)
        tp = np.load(row.thermodynamic_tensor_path)
        r = rp["radar_rate_mm_hr"][:, 32:96, 32:96]
        rm = rp["radar_valid_mask"][:, 32:96, 32:96]
        rl = (np.log1p(np.nan_to_num(r)) - rn["mean"]) / rn["std"]
        radar.append(np.stack([rl, rm], 1))
        raw = tp["thermodynamics"][:, :, 32:96, 32:96]
        valid = tp["thermodynamics_valid_mask"][:, :, 32:96, 32:96]
        tr = _transform(raw)
        norm = (tr - mean[None, :, None, None]) / std[None, :, None, None]
        thermo.append(np.concatenate([norm, valid], 1))
        target.append(rp["target_rate_mm_hr"][:, 32:96, 32:96])
        mask.append(rp["target_valid_mask"][:, 32:96, 32:96])
    r = torch.tensor(np.stack(radar), dtype=torch.float32, device=device)
    t = torch.tensor(np.stack(thermo), dtype=torch.float32, device=device)
    y = torch.tensor(np.stack(target), dtype=torch.float32, device=device)
    m = torch.tensor(np.stack(mask), dtype=torch.float32, device=device)
    model = RadarThermodynamicConvLSTM().to(device)
    model.load_radar_weights(
        torch.load("artifacts/stage_4b/a_plus_radar/model_best.pt", map_location=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    thresholds = torch.tensor([0.1, 1, 2.5, 5], device=device)[None, None, :, None, None]
    occurrence = (y[:, :, None] >= thresholds).float()
    logy = torch.log1p(torch.nan_to_num(y))
    losses = []
    gradients = {}
    for step in range(160):
        opt.zero_grad()
        logits, intensity = model(r, t)
        loss = (
            F.binary_cross_entropy_with_logits(logits, occurrence, reduction="none") * m[:, :, None]
        ).mean() + 0.25 * (
            F.smooth_l1_loss(F.softplus(intensity), logy, reduction="none") * m * (y >= 0.1)
        ).mean()
        loss.backward()
        if step == 2:
            groups = {
                "radar": (model.enc1, model.enc2, model.enc3, model.recurrent),
                "thermodynamic": (
                    model.thermo_enc1,
                    model.thermo_enc2,
                    model.thermo_enc3,
                    model.thermo_recurrent,
                ),
                "fusion": (model.thermo_to_radar,),
            }
            gradients = {
                k: sum(
                    float(p.grad.abs().sum())
                    for mod in mods
                    for p in mod.parameters()
                    if p.grad is not None
                )
                for k, mods in groups.items()
            }
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        base = model(r, t)[0]
        zero = model(r, torch.zeros_like(t))[0]
        shuffled = model(r, t.flip(0))[0]
        moisture = t.clone()
        moisture[:, :, 1] = 0
        moisture[:, :, 4] = 0
        instability = t.clone()
        instability[:, :, 2:4] = 0
        changes = {
            "all_zero": float((base - zero).abs().mean()),
            "shuffled": float((base - shuffled).abs().mean()),
            "moisture_zero": float((base - model(r, moisture)[0]).abs().mean()),
            "instability_zero": float((base - model(r, instability)[0]).abs().mean()),
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
        "all_paths_nonzero": all(v > 0 for v in gradients.values())
        and all(v > 0 for v in changes.values()),
        "parameter_breakdown": model.parameter_breakdown().__dict__,
    }
    out = Path("artifacts/stage_4c/tiny_overfit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tiny_overfit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
