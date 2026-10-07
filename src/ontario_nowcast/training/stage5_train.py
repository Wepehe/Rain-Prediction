"""Train the predeclared Stage 5 non-radar probe and frozen-A+ residual model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ..models.frozen_residual_fusion import (
    FrozenBaselineResidualFusion,
    NonRadarPredictabilityProbe,
)
from .stage5_data import Dataset, collate

A_PLUS = Path("artifacts/stage_4b/a_plus_radar/model_best.pt")
A_PLUS_SHA = "63e049c5ea75c3aa6af7b461fe6e9bca2db3c51198a667e71f02a7c6217efe70"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pos_weights(dataset: Dataset) -> list[float]:
    positive = np.zeros(4, dtype=np.float64)
    valid = 0.0
    for example in dataset:
        mask = example.target_mask.numpy() > 0
        valid += mask.sum()
        positive += (example.target_occurrence.numpy() * mask[:, None]).sum(axis=(0, 2, 3))
    return np.clip((valid - positive) / np.maximum(positive, 1), 1, 100).tolist()


def _h1_loss(logits, intensity, batch, weights):
    mask = batch["target_mask"][:, :, None] > 0
    raw = F.binary_cross_entropy_with_logits(
        logits,
        batch["target_occurrence"],
        pos_weight=weights.view(1, 1, 4, 1, 1),
        reduction="none",
    )
    occurrence = raw[mask.expand_as(raw)].mean()
    rainy = (batch["target_occurrence"][:, :, 0] > 0) & (batch["target_mask"] > 0)
    intensity_loss = F.smooth_l1_loss(
        F.softplus(intensity[rainy]), batch["target_intensity"][rainy]
    )
    return occurrence + 0.5 * intensity_loss


def _probe_loss(logits, batch, weight):
    mask = batch["target_mask"] > 0
    raw = F.binary_cross_entropy_with_logits(
        logits, batch["target_occurrence"][:, :, 0], pos_weight=weight, reduction="none"
    )
    return raw[mask].mean()


def _move(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train(kind: str) -> dict:
    if _sha(A_PLUS) != A_PLUS_SHA:
        raise RuntimeError("frozen A+ checkpoint hash changed")
    torch.manual_seed(5101 if kind == "probe" else 5102)
    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, dev_set = Dataset({"train"}), Dataset({"dev", "stage4_dev_repair"})
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True, collate_fn=collate
    )
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=4, collate_fn=collate)
    weights = torch.tensor(_pos_weights(train_set), dtype=torch.float32, device=device)
    model = NonRadarPredictabilityProbe() if kind == "probe" else FrozenBaselineResidualFusion()
    model = model.to(device)
    directory = {"probe": "nonradar_probe", "h1": "h1", "a_plus_only": "a_plus_only_control"}[kind]
    output = Path("artifacts/stage_5") / directory
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, 31):
        row = {"epoch": epoch}
        for phase, loader in (("train", train_loader), ("dev", dev_loader)):
            model.train(phase == "train")
            losses, residuals = [], []
            for raw_batch in loader:
                batch = _move(raw_batch, device)
                with torch.set_grad_enabled(phase == "train"):
                    if kind == "probe":
                        logits = model(batch["hrrr"], batch["goes"])
                        loss = _probe_loss(logits, batch, weights[0])
                    else:
                        hrrr = batch["hrrr"]
                        goes = batch["goes"]
                        if kind == "a_plus_only":
                            hrrr, goes = torch.zeros_like(hrrr), torch.zeros_like(goes)
                        logits, intensity, residual = model(
                            batch["base_logits"],
                            batch["base_intensity"],
                            batch["base_expected"],
                            hrrr,
                            goes,
                        )
                        loss = _h1_loss(logits, intensity, batch, weights)
                        residuals.append(float((4 * torch.tanh(residual[:, :, :4])).abs().mean()))
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                losses.append(float(loss.detach()))
            row[f"{phase}_loss"] = float(np.mean(losses))
            if residuals:
                row[f"{phase}_mean_abs_delta_logit"] = float(np.mean(residuals))
        history.append(row)
        scheduler.step(row["dev_loss"])
        print(f"[stage5-{kind}] epoch {epoch}: {row}", flush=True)
        if row["dev_loss"] < best - 1e-5:
            best, stale = row["dev_loss"], 0
            torch.save(model.state_dict(), output / "model_best.pt")
        else:
            stale += 1
            if stale >= 6:
                break
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    metadata = {
        "kind": kind,
        "device": str(device),
        "train_samples": len(train_set),
        "dev_samples": len(dev_set),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "radar_inputs_to_probe": False if kind == "probe" else "cached_A_plus_outputs_only",
        "nonradar_inputs_forced_zero": kind == "a_plus_only",
        "a_plus_trainable_parameters": 0,
        "occurrence_pos_weight_train_only": weights.cpu().tolist(),
        "epochs_ran": len(history),
        "best_dev_loss": best,
        "checkpoint_sha256": _sha(output / "model_best.pt"),
        "a_plus_checkpoint_bit_identical": _sha(A_PLUS) == A_PLUS_SHA,
        "protected_2023_holdout_opened": False,
    }
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("probe", "h1", "a_plus_only"))
    print(json.dumps(train(parser.parse_args().kind), indent=2))
