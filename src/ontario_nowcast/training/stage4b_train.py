"""Train Stage 4B radar+GOES C13 ablations."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..models.radar_goes import build_stage4b_model
from .stage3_loss import loss_kwargs_from_config, radar_multitask_loss

DEV_SPLITS = {"dev", "stage4_dev_repair"}
HOLDOUT_SPLITS = {"stage4_initiation_holdout"}
SMOKE_MODEL_LABEL = "radar_goes_pipeline_smoke_model"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class Stage4BExample:
    radar_inputs: Any
    goes_inputs: Any
    target_occurrence: Any
    target_intensity: Any
    target_mask: Any
    metadata: dict[str, Any]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {
        "radar": payload["radar_rate_mm_hr"].astype(np.float32),
        "radar_mask": payload["radar_valid_mask"].astype(np.float32),
        "goes": payload["goes"].astype(np.float32),
        "goes_mask": payload["goes_valid_mask"].astype(np.float32),
        "target": payload["target_rate_mm_hr"].astype(np.float32),
        "target_mask": payload["target_valid_mask"].astype(np.float32),
    }


def compute_goes_normalization(
    manifest: pd.DataFrame,
    *,
    goes_channel_indices: list[int],
) -> dict[str, Any]:
    """Compute train-only GOES per-channel normalization metadata."""
    count = np.zeros(len(goes_channel_indices), dtype=np.float64)
    total = np.zeros(len(goes_channel_indices), dtype=np.float64)
    total_sq = np.zeros(len(goes_channel_indices), dtype=np.float64)
    for row in manifest[manifest["split"].astype(str).eq("train")].itertuples(index=False):
        payload = _load_npz(Path(str(row.tensor_path)))
        values = payload["goes"][goes_channel_indices]
        mask = payload["goes_mask"][goes_channel_indices] > 0
        for index in range(len(goes_channel_indices)):
            valid = mask[index] & np.isfinite(values[index])
            if not np.any(valid):
                continue
            channel_values = values[index][valid].astype(np.float64)
            count[index] += channel_values.size
            total[index] += channel_values.sum()
            total_sq[index] += np.square(channel_values).sum()
    if np.any(count <= 0):
        raise ValueError("cannot compute GOES normalization; at least one channel has no train pixels")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1e-6)
    return {
        "source_split": "train",
        "goes_channel_indices": goes_channel_indices,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "count": [int(value) for value in count],
    }


def compute_occurrence_pos_weights(
    manifest: pd.DataFrame,
    thresholds_mm_hr: list[float],
) -> list[float]:
    """Derive multi-threshold BCE positive weights from train tensors only."""
    positive = np.zeros(len(thresholds_mm_hr), dtype=np.float64)
    valid_total = 0.0
    for row in manifest[manifest["split"].astype(str).eq("train")].itertuples(index=False):
        payload = _load_npz(Path(str(row.tensor_path)))
        target = payload["target"]
        valid = (payload["target_mask"] > 0) & np.isfinite(target)
        if not np.any(valid):
            continue
        values = target[valid]
        valid_total += float(values.size)
        for index, threshold in enumerate(thresholds_mm_hr):
            positive[index] += float(np.sum(values >= threshold))
    if valid_total <= 0:
        raise ValueError("no valid train target pixels for Stage 4B class weights")
    weights = []
    for value in positive:
        positives = max(float(value), 1.0)
        negatives = max(valid_total - positives, 1.0)
        weights.append(float(np.clip(negatives / positives, 1.0, 100.0)))
    return weights


class Stage4BRadarGoesDataset:
    """Dataset backed by materialized Stage 4 radar+GOES tensor files."""

    def __init__(
        self,
        manifest: pd.DataFrame | Path,
        *,
        splits: set[str],
        radar_normalization: dict[str, float],
        goes_normalization: dict[str, Any],
        goes_channel_indices: list[int],
        rain_threshold_mm_hr: float = 0.1,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install torch before Stage 4B learned training") from exc
        self.torch = torch
        table = pd.read_csv(manifest) if isinstance(manifest, Path) else manifest.copy()
        table = table[table["split"].astype(str).isin(splits)].copy()
        if table["split"].astype(str).isin(HOLDOUT_SPLITS).any():
            raise ValueError("final Stage 4 initiation holdout must not be used in Stage 4B training")
        self.table = table.reset_index(drop=True)
        self.radar_mean = float(radar_normalization["mean"])
        self.radar_std = max(float(radar_normalization["std"]), 1e-6)
        self.goes_channel_indices = goes_channel_indices
        self.goes_mean = np.asarray(goes_normalization["mean"], dtype=np.float32)
        self.goes_std = np.maximum(np.asarray(goes_normalization["std"], dtype=np.float32), 1e-6)
        self.rain_threshold_mm_hr = float(rain_threshold_mm_hr)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> Stage4BExample:
        row = self.table.iloc[index]
        payload = _load_npz(Path(str(row["tensor_path"])))
        radar = payload["radar"]
        radar_mask = payload["radar_mask"]
        radar_log = np.where(radar_mask > 0, np.log1p(np.nan_to_num(radar, nan=0.0)), 0.0)
        radar_norm = (radar_log - self.radar_mean) / self.radar_std
        radar_inputs = np.stack([radar_norm, radar_mask], axis=1).astype(np.float32)

        goes = payload["goes"][self.goes_channel_indices]
        goes_mask = payload["goes_mask"][self.goes_channel_indices]
        goes_norm = (np.nan_to_num(goes, nan=0.0) - self.goes_mean[:, None, None, None]) / (
            self.goes_std[:, None, None, None]
        )
        goes_values_time_major = np.moveaxis(goes_norm, 1, 0)
        goes_mask_time_major = np.moveaxis(goes_mask, 1, 0)
        goes_inputs = np.concatenate(
            [goes_values_time_major, goes_mask_time_major],
            axis=1,
        ).astype(np.float32)

        target = payload["target"]
        target_mask = payload["target_mask"]
        target_log = np.where(target_mask > 0, np.log1p(np.nan_to_num(target, nan=0.0)), 0.0)
        occurrence = ((target >= self.rain_threshold_mm_hr) & (target_mask > 0)).astype(np.float32)
        return Stage4BExample(
            radar_inputs=self.torch.from_numpy(radar_inputs),
            goes_inputs=self.torch.from_numpy(goes_inputs),
            target_occurrence=self.torch.from_numpy(occurrence),
            target_intensity=self.torch.from_numpy(target_log.astype(np.float32)),
            target_mask=self.torch.from_numpy(target_mask.astype(np.float32)),
            metadata=row.to_dict(),
        )


def collate_stage4b(examples: list[Stage4BExample]) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty Stage 4B batch")
    import torch

    return {
        "radar_inputs": torch.stack([example.radar_inputs for example in examples]),
        "goes_inputs": torch.stack([example.goes_inputs for example in examples]),
        "target_occurrence": torch.stack([example.target_occurrence for example in examples]),
        "target_intensity": torch.stack([example.target_intensity for example in examples]),
        "target_mask": torch.stack([example.target_mask for example in examples]),
        "metadata": [example.metadata for example in examples],
    }


def _batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _evaluate_loss(
    model: Any,
    loader: Any,
    device: Any,
    loss_kwargs: dict[str, Any],
) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            occurrence_logits, intensity_raw = model(batch["radar_inputs"], batch["goes_inputs"])
            loss, _ = radar_multitask_loss(
                occurrence_logits,
                intensity_raw,
                batch["target_occurrence"],
                batch["target_intensity"],
                batch["target_mask"],
                **loss_kwargs,
            )
            losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / len(losses)) if losses else float("nan")


def train_dev(
    model: Any,
    train_loader: Any,
    dev_loader: Any,
    device: Any,
    loss_kwargs: dict[str, Any],
    *,
    epochs: int,
    patience: int,
    learning_rate: float,
    output_dir: Path,
    progress_every_batches: int = 10,
) -> dict[str, Any]:
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
    )
    best_dev = float("inf")
    stale = 0
    rows = []
    best_path = output_dir / "model_best.pt"
    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            occurrence_logits, intensity_raw = model(batch["radar_inputs"], batch["goes_inputs"])
            loss, _ = radar_multitask_loss(
                occurrence_logits,
                intensity_raw,
                batch["target_occurrence"],
                batch["target_intensity"],
                batch["target_mask"],
                **loss_kwargs,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
                print(
                    f"[stage4b] epoch {epoch} batch {batch_index}/{len(train_loader)} "
                    f"loss={train_losses[-1]:.4f}",
                    flush=True,
                )
        dev_loss = _evaluate_loss(model, dev_loader, device, loss_kwargs)
        scheduler.step(dev_loss)
        train_loss = float(sum(train_losses) / len(train_losses))
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": float(time.time() - epoch_start),
            }
        )
        print(
            f"[stage4b] epoch {epoch}: train_loss={train_loss:.4f} "
            f"dev_loss={dev_loss:.4f}",
            flush=True,
        )
        if dev_loss < best_dev:
            best_dev = dev_loss
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
        torch.save(model.state_dict(), output_dir / "model_last.pt")
        pd.DataFrame(rows).to_csv(output_dir / "train_dev_curve.csv", index=False)
        (output_dir / "training_state.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "epochs_completed": epoch,
                    "best_dev_loss": best_dev,
                    "stale_epochs": stale,
                    "elapsed_seconds": float(time.time() - start),
                    "last_epoch": rows[-1],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if stale >= patience:
            break
    state = {
        "status": "complete",
        "epochs_completed": len(rows),
        "best_dev_loss": best_dev,
        "stale_epochs": stale,
        "elapsed_seconds": float(time.time() - start),
        "last_epoch": rows[-1] if rows else None,
        "best_checkpoint": best_path.as_posix(),
    }
    (output_dir / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def run_stage4b_training(
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
        held_out = manifest[manifest["split"].astype(str).isin(HOLDOUT_SPLITS)]
        manifest = manifest.drop(index=held_out.index).copy()
    goes_channel_indices = [int(value) for value in config["data"]["goes_channel_indices"]]
    radar_normalization = json.loads(
        Path(config["frozen_radar_control"]["normalization"]).read_text(encoding="utf-8")
    )
    radar_normalization = {
        "mean": float(radar_normalization["mean"]),
        "std": float(radar_normalization["std"]),
    }
    goes_normalization = compute_goes_normalization(
        manifest,
        goes_channel_indices=goes_channel_indices,
    )
    normalization = {
        "radar": {"source": "frozen_stage_3_1_train_only", **radar_normalization},
        "goes": goes_normalization,
    }
    (output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    train_dataset = Stage4BRadarGoesDataset(
        manifest,
        splits={"train"},
        radar_normalization=radar_normalization,
        goes_normalization=goes_normalization,
        goes_channel_indices=goes_channel_indices,
        rain_threshold_mm_hr=float(config["data"].get("rain_threshold_mm_hr", 0.1)),
    )
    dev_dataset = Stage4BRadarGoesDataset(
        manifest,
        splits=DEV_SPLITS,
        radar_normalization=radar_normalization,
        goes_normalization=goes_normalization,
        goes_channel_indices=goes_channel_indices,
        rain_threshold_mm_hr=float(config["data"].get("rain_threshold_mm_hr", 0.1)),
    )
    if len(train_dataset) == 0 or len(dev_dataset) == 0:
        raise ValueError("Stage 4B training requires non-empty train and DEV datasets")
    if "torch_num_threads" in config["training"]:
        torch.set_num_threads(int(config["training"]["torch_num_threads"]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_stage4b,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_stage4b,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_stage4b_model(config).to(device)
    checkpoint = torch.load(config["frozen_radar_control"]["checkpoint"], map_location="cpu")
    loaded_keys = model.load_radar_control_weights(checkpoint)
    model = model.to(device)
    thresholds = [float(value) for value in config["model"]["occurrence_thresholds_mm_hr"]]
    if bool(config["loss"].get("class_weights_from_train_only", False)):
        config["loss"]["occurrence_pos_weight"] = compute_occurrence_pos_weights(
            manifest,
            thresholds,
        )
    loss_kwargs = loss_kwargs_from_config(config["loss"], config["model"])
    breakdown = model.parameter_breakdown()
    metadata: dict[str, Any] = {
        "config": config_path.as_posix(),
        "tensor_manifest": tensor_manifest.as_posix(),
        "variant": config["name"],
        "pipeline_smoke_model_label": SMOKE_MODEL_LABEL,
        "uses_pipeline_smoke_model_for_comparison": False,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "splits": {
            "train": len(train_dataset),
            "dev_selection": len(dev_dataset),
            "excluded_final_initiation_holdout": int(
                (pd.read_csv(tensor_manifest)["split"].astype(str).isin(HOLDOUT_SPLITS)).sum()
            ),
        },
        "goes_channel_indices": goes_channel_indices,
        "goes_channel_names": config["data"]["goes_channel_names"],
        "radar_control_checkpoint_loaded": config["frozen_radar_control"]["checkpoint"],
        "loaded_radar_control_keys": len(loaded_keys),
        "parameter_breakdown": breakdown.__dict__,
        "capacity_control_decision": config.get("capacity_control", {}),
        "loss_kwargs": {
            key: value for key, value in loss_kwargs.items() if key != "occurrence_pos_weight"
        },
        "occurrence_pos_weight_train_only": loss_kwargs.get("occurrence_pos_weight"),
        "normalization": normalization,
        "final_holdout_scoring": "not_evaluated_during_development",
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tensor-manifest",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage4b_training(
                args.config,
                output_dir=args.output_dir,
                tensor_manifest=args.tensor_manifest,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
