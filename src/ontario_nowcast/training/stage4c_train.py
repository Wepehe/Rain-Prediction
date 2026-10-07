"""Train C1 on frozen TRAIN/DEV tensors without opening the Stage C holdout."""

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

from ..models.radar_thermodynamics import RadarThermodynamicConvLSTM
from .stage3_loss import loss_kwargs_from_config
from .stage4b_train import compute_occurrence_pos_weights, train_dev
from .stage4c_prepare import _transform


@dataclass(frozen=True)
class Example:
    radar_inputs: Any
    thermo_inputs: Any
    target_occurrence: Any
    target_intensity: Any
    target_mask: Any
    metadata: dict[str, Any]


class Dataset:
    def __init__(
        self,
        table: pd.DataFrame,
        splits: set[str],
        radar_norm: dict[str, float],
        thermo_norm: dict[str, Any],
    ):
        self.table = table[table.split.isin(splits)].reset_index(drop=True)
        self.rn = radar_norm
        self.tn = thermo_norm

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        rp = np.load(row.radar_tensor_path)
        tp = np.load(row.thermodynamic_tensor_path)
        r = rp["radar_rate_mm_hr"]
        rm = rp["radar_valid_mask"].astype(np.float32)
        radar = np.stack(
            [(np.log1p(np.nan_to_num(r)) - self.rn["mean"]) / self.rn["std"], rm], 1
        ).astype(np.float32)
        raw = tp["thermodynamics"]
        tm = tp["thermodynamics_valid_mask"].astype(np.float32)
        transformed = _transform(raw)
        mean = np.asarray(self.tn["mean"])[None, :, None, None]
        std = np.asarray(self.tn["std"])[None, :, None, None]
        thermo = np.concatenate([(transformed - mean) / std, tm], 1).astype(np.float32)
        target = rp["target_rate_mm_hr"].astype(np.float32)
        mask = rp["target_valid_mask"].astype(np.float32)
        return Example(
            torch.from_numpy(radar),
            torch.from_numpy(thermo),
            torch.from_numpy(((target >= 0.1) & (mask > 0)).astype(np.float32)),
            torch.from_numpy(np.log1p(np.nan_to_num(target)).astype(np.float32)),
            torch.from_numpy(mask),
            row.to_dict(),
        )


def collate(items: list[Example]) -> dict[str, Any]:
    return {
        "radar_inputs": torch.stack([x.radar_inputs for x in items]),
        "goes_inputs": torch.stack([x.thermo_inputs for x in items]),
        "target_occurrence": torch.stack([x.target_occurrence for x in items]),
        "target_intensity": torch.stack([x.target_intensity for x in items]),
        "target_mask": torch.stack([x.target_mask for x in items]),
        "metadata": [x.metadata for x in items],
    }


class Adapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, radar_inputs, thermo_inputs):
        return self.model(radar_inputs, thermo_inputs)


def run(config_path: Path = Path("configs/experiments/stage_4c_c1.yaml")) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text())
    out = Path("artifacts/stage_4c/c1")
    out.mkdir(parents=True, exist_ok=True)
    radar = pd.read_csv(cfg["data"]["radar_manifest"])
    thermo = pd.read_csv(cfg["data"]["thermodynamic_manifest"])
    table = thermo.copy()
    table = table[table.split.isin({"train", "dev", "stage4_dev_repair"})].copy()
    assert not table.split.str.contains("holdout").any()
    rn = json.loads(Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text())[
        "normalization"
    ]["radar"]
    tn = json.loads(
        Path("artifacts/stage_4c/gate/thermodynamic_normalization_train_only.json").read_text()
    )
    train = Dataset(table, {"train"}, rn, tn)
    dev = Dataset(table, {"dev", "stage4_dev_repair"}, rn, tn)
    torch.set_num_threads(cfg["training"]["torch_num_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarThermodynamicConvLSTM().to(device)
    loaded = model.load_radar_weights(
        torch.load("artifacts/stage_4b/a_plus_radar/model_best.pt", map_location=device)
    )
    thresholds = cfg["model"]["occurrence_thresholds_mm_hr"]
    weights = compute_occurrence_pos_weights(radar[radar.split.eq("train")], thresholds)
    cfg["loss"]["occurrence_pos_weight"] = weights
    loss = loss_kwargs_from_config(cfg["loss"], cfg["model"])
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=4, shuffle=True, collate_fn=collate
    )
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=4, collate_fn=collate)
    metadata = {
        "config": config_path.as_posix(),
        "device": str(device),
        "train_samples": len(train),
        "dev_samples": len(dev),
        "holdout_opened": False,
        "parameter_breakdown": model.parameter_breakdown().__dict__,
        "loaded_a_plus_keys": loaded,
        "normalization_hash": hashlib.sha256(
            Path("artifacts/stage_4c/gate/thermodynamic_normalization_train_only.json").read_bytes()
        ).hexdigest(),
        "occurrence_pos_weight": weights,
    }
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    state = train_dev(
        model,
        train_loader,
        dev_loader,
        device,
        loss,
        epochs=30,
        patience=6,
        learning_rate=0.0003,
        output_dir=out,
        progress_every_batches=4,
    )
    metadata["train_dev"] = state
    metadata["best_checkpoint_sha256"] = hashlib.sha256(
        (out / "model_best.pt").read_bytes()
    ).hexdigest()
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
