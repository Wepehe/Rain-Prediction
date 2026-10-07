"""Training entry point for the Stage 3 radar-only learned experiment."""

from __future__ import annotations

import argparse
import json
import time
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..models.radar_convlstm import build_radar_model, count_parameters
from .stage3_data import (
    Stage3RadarDataset,
    _load_event,
    build_sample_manifest,
    collate_stage3,
    compute_train_normalization,
)
from .stage3_loss import loss_kwargs_from_config, radar_multitask_loss


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_manifest(config_path: Path, manifest_path: Path) -> pd.DataFrame:
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return build_sample_manifest(config_path, output_path=manifest_path)


def _make_loaders(
    manifest_path: Path,
    normalization: dict[str, float],
    *,
    batch_size: int,
    preload_events: bool = False,
) -> tuple[Any, Any, Any]:
    import torch

    train = Stage3RadarDataset(
        manifest_path, split="train", normalization=normalization, preload=preload_events
    )
    dev = Stage3RadarDataset(
        manifest_path, split="dev", normalization=normalization, preload=preload_events
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, collate_fn=collate_stage3
    )
    dev_loader = torch.utils.data.DataLoader(
        dev, batch_size=batch_size, shuffle=False, collate_fn=collate_stage3
    )
    return train_loader, dev_loader, train


def _batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _loss_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return loss_kwargs_from_config(config["loss"], config["model"])


def compute_train_occurrence_pos_weights(
    manifest_path: Path,
    normalization: dict[str, float],
    thresholds_mm_hr: list[float],
) -> list[float]:
    """Derive multi-threshold BCE positive weights from training data only."""
    del normalization
    manifest = pd.read_csv(manifest_path)
    positive = pd.Series(0.0, index=range(len(thresholds_mm_hr)), dtype=float)
    valid_total = 0.0
    train_rows = manifest[manifest["split"] == "train"]
    for event_id, event_rows in train_rows.groupby("event_id"):
        print(
            f"[training] class weights scanning {event_id}: {len(event_rows)} samples",
            flush=True,
        )
        rates, _ = _load_event(str(event_id), Path("data"))
        for row in event_rows.itertuples(index=False):
            target = rates[
                int(row.target_start_index) : int(row.target_end_index) + 1,
                int(row.y0) : int(row.y0) + int(row.tile_pixels),
                int(row.x0) : int(row.x0) + int(row.tile_pixels),
            ]
            finite = np.isfinite(target)
            if not np.any(finite):
                continue
            values = target[finite]
            valid_total += float(values.size)
            for threshold_index, threshold in enumerate(thresholds_mm_hr):
                positive.iloc[threshold_index] += float(np.sum(values >= threshold))
    if valid_total <= 0:
        raise ValueError("no valid training target pixels found for class weights")
    weights = []
    for value in positive:
        positives = max(float(value), 1.0)
        negatives = max(valid_total - positives, 1.0)
        weights.append(float(np.clip(negatives / positives, 1.0, 100.0)))
    return weights


def _evaluate_loss(
    model: Any,
    loader: Any,
    device: Any,
    loss_kwargs: dict[str, Any],
    *,
    max_batches: int | None = None,
) -> float:
    import torch

    losses = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            batch = _batch_to_device(batch, device)
            occurrence_logits, intensity_raw = model(batch["inputs"])
            loss, _ = radar_multitask_loss(
                occurrence_logits,
                intensity_raw,
                batch["target_occurrence"],
                batch["target_intensity"],
                batch["target_mask"],
                **loss_kwargs,
            )
            losses.append(float(loss.detach().cpu()))
            if max_batches is not None and index >= max_batches:
                break
    return float(sum(losses) / len(losses)) if losses else float("nan")


def tiny_overfit(
    model: Any,
    train_loader: Any,
    device: Any,
    loss_kwargs: dict[str, Any],
    *,
    batches: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    selected = []
    for index, batch in enumerate(train_loader):
        selected.append(_batch_to_device(batch, device))
        if index + 1 >= batches:
            break
    if not selected:
        raise ValueError("tiny overfit requested but training loader is empty")
    losses = []
    gradient_norms = []
    model.train()
    for step, batch in zip(range(steps), cycle(selected), strict=False):
        optimizer.zero_grad(set_to_none=True)
        occurrence_logits, intensity_raw = model(batch["inputs"])
        loss, parts = radar_multitask_loss(
            occurrence_logits,
            intensity_raw,
            batch["target_occurrence"],
            batch["target_intensity"],
            batch["target_mask"],
            **loss_kwargs,
        )
        loss.backward()
        squared_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_norm += float(parameter.grad.detach().pow(2).sum().cpu())
        gradient_norms.append(squared_norm**0.5)
        optimizer.step()
        losses.append({"step": step + 1, **parts})
        if step == 0 or (step + 1) % 5 == 0:
            print(f"[tiny-overfit] step {step + 1}/{steps}: loss={parts['total_loss']:.4f}", flush=True)
    with torch.no_grad():
        batch = selected[0]
        occurrence_logits, intensity_raw = model(batch["inputs"])
        probability = torch.sigmoid(occurrence_logits)
        if probability.ndim == 5:
            probability = probability[:, :, 0]
        expected_rate = torch.expm1(torch.nn.functional.softplus(intensity_raw)) * probability
        observed_rate = torch.expm1(batch["target_intensity"])
        valid = batch["target_mask"] > 0
        rainy = valid & (batch["target_occurrence"] > 0)
        dry = valid & (batch["target_occurrence"] <= 0)
        expected_mae = torch.mean(torch.abs(expected_rate[valid] - observed_rate[valid]))
        wet_expected_mae = (
            torch.mean(torch.abs(expected_rate[rainy] - observed_rate[rainy]))
            if torch.any(rainy)
            else expected_mae * float("nan")
        )
    return {
        "steps": steps,
        "batches": batches,
        "initial_loss": losses[0]["total_loss"],
        "final_loss": losses[-1]["total_loss"],
        "loss_decreased": losses[-1]["total_loss"] < losses[0]["total_loss"],
        "initial_to_final_loss_ratio": losses[-1]["total_loss"] / max(losses[0]["total_loss"], 1e-9),
        "mean_gradient_norm": float(sum(gradient_norms) / len(gradient_norms)),
        "final_probability_mean_dry_pixels": float(probability[dry].mean().detach().cpu())
        if torch.any(dry)
        else None,
        "final_probability_mean_rainy_pixels": float(probability[rainy].mean().detach().cpu())
        if torch.any(rainy)
        else None,
        "final_expected_rate_mae_mm_hr": float(expected_mae.detach().cpu()),
        "final_wet_pixel_expected_rate_mae_mm_hr": float(wet_expected_mae.detach().cpu()),
        "finite_target_mask_fraction": float(valid.float().mean().detach().cpu()),
        "losses": losses,
    }


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
    max_train_batches_per_epoch: int | None = None,
    max_dev_batches: int | None = None,
    progress_every_batches: int = 50,
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
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            occurrence_logits, intensity_raw = model(batch["inputs"])
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
                    f"[train-dev] epoch {epoch} batch {batch_index}/{len(train_loader)} "
                    f"loss={train_losses[-1]:.4f}",
                    flush=True,
                )
            if (
                max_train_batches_per_epoch is not None
                and batch_index >= max_train_batches_per_epoch
            ):
                break
        dev_loss = _evaluate_loss(model, dev_loader, device, loss_kwargs, max_batches=max_dev_batches)
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
            f"[train-dev] epoch {epoch}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}",
            flush=True,
        )
        if dev_loss < best_dev:
            best_dev = dev_loss
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
        pd.DataFrame(rows).to_csv(output_dir / "train_dev_curve.csv", index=False)
        torch.save(model.state_dict(), output_dir / "model_last.pt")
        (output_dir / "training_state.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "epochs_completed": epoch,
                    "best_dev_loss": best_dev,
                    "stale_epochs": stale,
                    "elapsed_seconds": float(time.time() - start_time),
                    "last_epoch": rows[-1],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if stale >= patience:
            break
    pd.DataFrame(rows).to_csv(output_dir / "train_dev_curve.csv", index=False)
    (output_dir / "training_state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "epochs_completed": len(rows),
                "best_dev_loss": best_dev,
                "stale_epochs": stale,
                "elapsed_seconds": float(time.time() - start_time),
                "last_epoch": rows[-1] if rows else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"best_dev_loss": best_dev, "epochs_ran": len(rows), "best_checkpoint": best_path.as_posix()}


def run_training(
    config_path: Path,
    *,
    manifest_path: Path = Path("artifacts/stage_3/sample_manifest.csv"),
    output_dir: Path = Path("artifacts/stage_3"),
    run_tiny_overfit: bool = False,
    run_train_dev: bool = False,
) -> dict[str, Any]:
    import torch

    config = _load_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _ensure_manifest(config_path, manifest_path)
    normalization_path = output_dir / "normalization.json"
    if normalization_path.exists():
        cached_normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        train_sample_count = int((manifest["split"] == "train").sum())
        if (
            cached_normalization.get("source_split") != "train"
            or cached_normalization.get("train_samples") != train_sample_count
        ):
            normalization = compute_train_normalization(manifest)
            normalization_path.write_text(
                json.dumps(
                    {"source_split": "train", "train_samples": train_sample_count, **normalization},
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("[training] recomputed train-only normalization", flush=True)
        else:
            normalization = {
                "mean": float(cached_normalization["mean"]),
                "std": float(cached_normalization["std"]),
            }
            print("[training] reused train-only normalization", flush=True)
    else:
        normalization = compute_train_normalization(manifest)
        normalization_path.write_text(
            json.dumps(
                {
                    "source_split": "train",
                    "train_samples": int((manifest["split"] == "train").sum()),
                    **normalization,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if not normalization_path.exists():
        raise ValueError("normalization file was not written")
    cached_normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if cached_normalization.get("source_split") != "train":
        raise ValueError("cached normalization is not marked as train-only")
    if "torch_num_threads" in config["training"]:
        torch.set_num_threads(int(config["training"]["torch_num_threads"]))
        print(f"[training] torch_num_threads={torch.get_num_threads()}", flush=True)
    train_loader, dev_loader, train_dataset = _make_loaders(
        manifest_path,
        normalization,
        batch_size=int(config["training"]["batch_size"]),
        preload_events=bool(config["training"].get("preload_events", False)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(config["loss"].get("class_weights_from_train_only", False)):
        thresholds = [float(value) for value in config["model"].get("occurrence_thresholds_mm_hr", [0.1])]
        config["loss"]["occurrence_pos_weight"] = compute_train_occurrence_pos_weights(
            manifest_path, normalization, thresholds
        )
        print(
            f"[training] train-only occurrence pos_weight={config['loss']['occurrence_pos_weight']}",
            flush=True,
        )
    model = build_radar_model(config).to(device)
    loss_kwargs = _loss_kwargs(config)
    metadata: dict[str, Any] = {
        "config": config_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "parameter_count": count_parameters(model),
        "architecture": str(config["model"].get("architecture", "minimal_convlstm")),
        "loss_kwargs": {
            key: value for key, value in loss_kwargs.items() if key != "occurrence_pos_weight"
        },
        "occurrence_pos_weight_train_only": loss_kwargs.get("occurrence_pos_weight"),
        "train_samples": int((manifest["split"] == "train").sum()),
        "dev_samples": int((manifest["split"] == "dev").sum()),
        "test_samples_frozen_unused": int((manifest["split"] == "test").sum()),
        "existing_test_samples_frozen_unused": int((manifest["split"] == "existing_test").sum()),
        "fresh_holdout_samples_frozen_unused": int((manifest["split"] == "fresh_holdout").sum()),
        "normalization": {"source_split": "train", **normalization},
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if run_tiny_overfit:
        tiny_batch_size = int(
            config["training"].get("tiny_overfit_batch_size", config["training"]["batch_size"])
        )
        tiny_train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=tiny_batch_size,
            shuffle=True,
            collate_fn=collate_stage3,
        )
        metadata["tiny_overfit"] = tiny_overfit(
            model,
            tiny_train_loader,
            device,
            loss_kwargs,
            batches=int(config["training"]["tiny_overfit_batches"]),
            steps=int(config["training"]["tiny_overfit_steps"]),
            learning_rate=float(config["training"]["learning_rate"]),
        )
        (output_dir / "tiny_overfit_summary.json").write_text(
            json.dumps(metadata["tiny_overfit"], indent=2),
            encoding="utf-8",
        )
        torch.save(model.state_dict(), output_dir / "tiny_overfit_model.pt")
        if run_train_dev:
            model = build_radar_model(config).to(device)
    if run_train_dev:
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
            max_train_batches_per_epoch=config["training"].get("max_train_batches_per_epoch"),
            max_dev_batches=config["training"].get("max_dev_batches"),
            progress_every_batches=int(config["training"].get("progress_every_batches", 50)),
        )
    if torch.cuda.is_available():
        metadata["gpu_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    else:
        metadata["gpu_peak_memory_bytes"] = None
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/stage_3_radar_only.yaml")
    )
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/stage_3/sample_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_3"))
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--train-dev", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_training(
                args.config,
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                run_tiny_overfit=args.tiny_overfit,
                run_train_dev=args.train_dev,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
