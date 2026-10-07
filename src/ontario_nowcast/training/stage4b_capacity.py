"""Radar-only A+ capacity control for Stage 4B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..models.radar_convlstm import build_radar_model, count_parameters
from .stage3_loss import loss_kwargs_from_config
from .stage3_train import train_dev
from .stage4b_train import (
    HOLDOUT_SPLITS,
    Stage4BRadarGoesDataset,
    collate_stage4b,
    compute_occurrence_pos_weights,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class Stage4BRadarOnlyTensorDataset(Stage4BRadarGoesDataset):
    """Stage 4B tensor dataset exposing only radar inputs to the Stage 3.1 trainer."""

    def __getitem__(self, index: int) -> Any:
        example = super().__getitem__(index)
        return type(example)(
            radar_inputs=example.radar_inputs,
            goes_inputs=example.goes_inputs,
            target_occurrence=example.target_occurrence,
            target_intensity=example.target_intensity,
            target_mask=example.target_mask,
            metadata=example.metadata,
        )


def _collate_radar_only(examples: list[Any]) -> dict[str, Any]:
    batch = collate_stage4b(examples)
    return {
        "inputs": batch["radar_inputs"],
        "target_occurrence": batch["target_occurrence"],
        "target_intensity": batch["target_intensity"],
        "target_mask": batch["target_mask"],
        "metadata": batch["metadata"],
    }


def run_capacity_control(
    config_path: Path,
    *,
    output_dir: Path,
    tensor_manifest: Path = Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
) -> dict[str, Any]:
    import torch

    config = _load_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(tensor_manifest)
    if manifest["split"].astype(str).isin(HOLDOUT_SPLITS).any():
        manifest = manifest[~manifest["split"].astype(str).isin(HOLDOUT_SPLITS)].copy()
    radar_normalization_raw = json.loads(
        Path(config["frozen_radar_control"]["normalization"]).read_text(encoding="utf-8")
    )
    radar_normalization = {
        "mean": float(radar_normalization_raw["mean"]),
        "std": float(radar_normalization_raw["std"]),
    }
    dummy_goes_normalization = {"mean": [0.0], "std": [1.0], "goes_channel_indices": [0]}
    train_dataset = Stage4BRadarOnlyTensorDataset(
        manifest,
        splits={"train"},
        radar_normalization=radar_normalization,
        goes_normalization=dummy_goes_normalization,
        goes_channel_indices=[0],
        rain_threshold_mm_hr=float(config["data"].get("rain_threshold_mm_hr", 0.1)),
    )
    dev_dataset = Stage4BRadarOnlyTensorDataset(
        manifest,
        splits={"dev", "stage4_dev_repair"},
        radar_normalization=radar_normalization,
        goes_normalization=dummy_goes_normalization,
        goes_channel_indices=[0],
        rain_threshold_mm_hr=float(config["data"].get("rain_threshold_mm_hr", 0.1)),
    )
    if "torch_num_threads" in config["training"]:
        torch.set_num_threads(int(config["training"]["torch_num_threads"]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=_collate_radar_only,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=_collate_radar_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(config["loss"].get("class_weights_from_train_only", False)):
        config["loss"]["occurrence_pos_weight"] = compute_occurrence_pos_weights(
            manifest, [float(value) for value in config["model"]["occurrence_thresholds_mm_hr"]]
        )
    model = build_radar_model({"model": config["model"], "data": config["data"]}).to(device)
    loss_kwargs = loss_kwargs_from_config(config["loss"], config["model"])
    metadata: dict[str, Any] = {
        "config": config_path.as_posix(),
        "tensor_manifest": tensor_manifest.as_posix(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "parameter_count": count_parameters(model),
        "train_samples": len(train_dataset),
        "dev_selection_samples": len(dev_dataset),
        "normalization": {"radar": {"source": "frozen_stage_3_1_train_only", **radar_normalization}},
        "occurrence_pos_weight_train_only": loss_kwargs.get("occurrence_pos_weight"),
        "final_holdout_scoring": "not_evaluated",
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["train_dev"] = train_dev(
        model,
        train_loader,
        dev_loader,
        device,
        loss_kwargs,
        epochs=int(config["training"]["max_epochs"]),
        patience=int(config["training"]["early_stopping_patience"]),
        learning_rate=float(config["training"]["learning_rate"]),
        output_dir=output_dir,
        progress_every_batches=int(config["training"].get("progress_every_batches", 10)),
    )
    if torch.cuda.is_available():
        metadata["gpu_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    else:
        metadata["gpu_peak_memory_bytes"] = None
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/stage_4b_a_plus_radar_capacity.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4b/a_plus_radar"))
    parser.add_argument(
        "--tensor-manifest",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_capacity_control(
                args.config,
                output_dir=args.output_dir,
                tensor_manifest=args.tensor_manifest,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
