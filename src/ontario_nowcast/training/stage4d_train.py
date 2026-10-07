"""Train D1 on the frozen TRAIN/DEV dynamics tensors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ..models.radar_dynamics import RadarDynamicsConvLSTM
from .stage3_loss import loss_kwargs_from_config
from .stage4b_train import compute_occurrence_pos_weights, train_dev

ALLOWED_SPLITS = {"train", "dev", "stage4_dev_repair"}


@dataclass(frozen=True)
class Example:
    radar_inputs: Any
    dynamics_inputs: Any
    target_occurrence: Any
    target_intensity: Any
    target_mask: Any
    metadata: dict[str, Any]


class Dataset:
    def __init__(self, table, splits, radar_norm, dynamics_norm):
        self.table = table[table.split.isin(splits)].reset_index(drop=True)
        self.radar_norm = radar_norm
        self.dynamics_norm = dynamics_norm

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        radar_payload = np.load(row.radar_tensor_path)
        dynamics_payload = np.load(row.dynamics_tensor_path)
        rate = radar_payload["radar_rate_mm_hr"]
        radar_mask = radar_payload["radar_valid_mask"].astype(np.float32)
        radar = np.stack(
            [
                (np.log1p(np.nan_to_num(rate)) - self.radar_norm["mean"]) / self.radar_norm["std"],
                radar_mask,
            ],
            1,
        ).astype(np.float32)
        raw = dynamics_payload["dynamics"]
        dynamics_mask = dynamics_payload["dynamics_valid_mask"].astype(np.float32)
        mean = np.asarray(self.dynamics_norm["mean"])[None, :, None, None]
        std = np.asarray(self.dynamics_norm["std"])[None, :, None, None]
        dynamics = np.concatenate([(raw - mean) / std, dynamics_mask], 1).astype(np.float32)
        target = radar_payload["target_rate_mm_hr"].astype(np.float32)
        target_mask = radar_payload["target_valid_mask"].astype(np.float32)
        return Example(
            torch.from_numpy(radar),
            torch.from_numpy(dynamics),
            torch.from_numpy(((target >= 0.1) & (target_mask > 0)).astype(np.float32)),
            torch.from_numpy(np.log1p(np.nan_to_num(target)).astype(np.float32)),
            torch.from_numpy(target_mask),
            row.to_dict(),
        )


def collate(items):
    return {
        "radar_inputs": torch.stack([item.radar_inputs for item in items]),
        "goes_inputs": torch.stack([item.dynamics_inputs for item in items]),
        "target_occurrence": torch.stack([item.target_occurrence for item in items]),
        "target_intensity": torch.stack([item.target_intensity for item in items]),
        "target_mask": torch.stack([item.target_mask for item in items]),
        "metadata": [item.metadata for item in items],
    }


def run(config_path: Path = Path("configs/experiments/stage_4d_d1.yaml")) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    output = Path("artifacts/stage_4d/d1")
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv("artifacts/stage_4d/tensors/stage4d_dynamics_tensor_manifest.csv")
    if not set(table.split.unique()) <= ALLOWED_SPLITS or len(table) != 72:
        raise RuntimeError("D1 training requires exactly 72 TRAIN/DEV tensors")
    radar_norm = json.loads(
        Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text()
    )["normalization"]["radar"]
    dynamics_norm_path = Path("artifacts/stage_4d/gate/dynamics_normalization_train_only.json")
    dynamics_norm = json.loads(dynamics_norm_path.read_text())
    train = Dataset(table, {"train"}, radar_norm, dynamics_norm)
    dev = Dataset(table, {"dev", "stage4_dev_repair"}, radar_norm, dynamics_norm)
    torch.set_num_threads(config["training"]["torch_num_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarDynamicsConvLSTM().to(device)
    loaded = model.load_radar_weights(
        torch.load("artifacts/stage_4b/a_plus_radar/model_best.pt", map_location=device)
    )
    thresholds = config["model"]["occurrence_thresholds_mm_hr"]
    weights = compute_occurrence_pos_weights(
        pd.read_csv(
            "artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"
        ).query("split == 'train'"),
        thresholds,
    )
    config["loss"]["occurrence_pos_weight"] = weights
    loss = loss_kwargs_from_config(config["loss"], config["model"])
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=4, shuffle=True, collate_fn=collate
    )
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=4, collate_fn=collate)
    metadata = {
        "config": config_path.as_posix(),
        "device": str(device),
        "train_samples": len(train),
        "dev_samples": len(dev),
        "protected_2023_holdout_opened": False,
        "parameter_breakdown": model.parameter_breakdown().__dict__,
        "loaded_a_plus_keys": loaded,
        "normalization_hash": hashlib.sha256(dynamics_norm_path.read_bytes()).hexdigest(),
        "occurrence_pos_weight": weights,
    }
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    state = train_dev(
        model,
        train_loader,
        dev_loader,
        device,
        loss,
        epochs=config["training"]["max_epochs"],
        patience=config["training"]["early_stopping_patience"],
        learning_rate=config["training"]["learning_rate"],
        output_dir=output,
        progress_every_batches=config["training"]["progress_every_batches"],
    )
    metadata["train_dev"] = state
    metadata["best_checkpoint_sha256"] = hashlib.sha256(
        (output / "model_best.pt").read_bytes()
    ).hexdigest()
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
