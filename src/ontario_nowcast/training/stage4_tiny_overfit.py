"""Tiny radar+GOES overfit gate for Stage 4 before Ablation B training."""

from __future__ import annotations

import argparse
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stage3_loss import radar_multitask_loss


def _load_tensor(path: Path) -> dict[str, np.ndarray | str]:
    payload = np.load(path)
    return {
        "radar": payload["radar_rate_mm_hr"].astype(np.float32),
        "radar_mask": payload["radar_valid_mask"].astype(np.float32),
        "goes": payload["goes"].astype(np.float32),
        "goes_mask": payload["goes_valid_mask"].astype(np.float32),
        "target": payload["target_rate_mm_hr"].astype(np.float32),
        "target_mask": payload["target_valid_mask"].astype(np.float32),
        "path": path.as_posix(),
    }


def _prepare_inputs(records: list[dict[str, np.ndarray | str]]) -> dict[str, Any]:
    import torch

    radar = np.stack([np.asarray(record["radar"]) for record in records])
    radar_mask = np.stack([np.asarray(record["radar_mask"]) for record in records])
    goes = np.stack([np.asarray(record["goes"]) for record in records])
    goes_mask = np.stack([np.asarray(record["goes_mask"]) for record in records])
    target = np.stack([np.asarray(record["target"]) for record in records])
    target_mask = np.stack([np.asarray(record["target_mask"]) for record in records])

    radar_log = np.log1p(np.nan_to_num(radar, nan=0.0))
    radar_mean = float(radar_log[radar_mask > 0].mean()) if np.any(radar_mask > 0) else 0.0
    radar_std = float(radar_log[radar_mask > 0].std()) if np.any(radar_mask > 0) else 1.0
    radar_norm = (radar_log - radar_mean) / max(radar_std, 1e-6)

    goes_values = np.nan_to_num(goes, nan=0.0)
    raw_c13 = goes_values[:, 0]
    cooling = goes_values[:, 1:]
    raw_norm = (raw_c13 - 260.0) / 30.0
    cooling_norm = cooling / 20.0
    goes_norm = np.concatenate([raw_norm[:, None], cooling_norm], axis=1)

    target_mask_bool = target_mask > 0
    target_log = np.where(target_mask_bool, np.log1p(np.nan_to_num(target, nan=0.0)), 0.0)
    target_occurrence = ((target >= 0.1) & target_mask_bool).astype(np.float32)

    return {
        "radar_input": torch.from_numpy(
            np.concatenate([radar_norm, radar_mask], axis=1).astype(np.float32)
        ),
        "goes_input": torch.from_numpy(
            np.concatenate(
                [goes_norm.reshape(goes_norm.shape[0], -1, *goes_norm.shape[-2:]), goes_mask.reshape(goes_mask.shape[0], -1, *goes_mask.shape[-2:])],
                axis=1,
            ).astype(np.float32)
        ),
        "target_occurrence": torch.from_numpy(target_occurrence),
        "target_intensity": torch.from_numpy(target_log.astype(np.float32)),
        "target_mask": torch.from_numpy(target_mask.astype(np.float32)),
        "diagnostics": {
            "radar_input_std": float(np.std(radar_norm)),
            "goes_raw_c13_std": float(np.std(raw_c13)),
            "goes_cooling_std": float(np.std(cooling)),
            "radar_mask_valid_fraction": float(np.mean(radar_mask > 0)),
            "goes_mask_valid_fraction": float(np.mean(goes_mask > 0)),
            "target_mask_valid_fraction": float(np.mean(target_mask > 0)),
            "target_rain_fraction": float(np.mean(target_occurrence[target_mask_bool]))
            if np.any(target_mask_bool)
            else float("nan"),
        },
    }


class TinyRadarGoesFusion:
    """Namespace wrapper around the torch module builder to keep imports lazy."""

    @staticmethod
    def build(radar_channels: int, goes_channels: int, target_frames: int) -> Any:
        import torch

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.radar_branch = torch.nn.Sequential(
                    torch.nn.Conv2d(radar_channels, 16, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                )
                self.satellite_branch = torch.nn.Sequential(
                    torch.nn.Conv2d(goes_channels, 16, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                )
                self.fusion = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, target_frames * 2, kernel_size=1),
                )
                self.target_frames = target_frames

            def forward(self, radar_input: Any, goes_input: Any) -> tuple[Any, Any]:
                radar_features = self.radar_branch(radar_input)
                satellite_features = self.satellite_branch(goes_input)
                output = self.fusion(torch.cat([radar_features, satellite_features], dim=1))
                occurrence_logits, intensity_raw = torch.split(output, self.target_frames, dim=1)
                return occurrence_logits, intensity_raw

        return _Model()


def _grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    return total**0.5


def run_tiny_overfit(
    tensor_manifest: Path,
    output_dir: Path,
    *,
    samples: int = 2,
    steps: int = 80,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    import torch

    manifest = pd.read_csv(tensor_manifest)
    train = manifest[manifest["split"].astype(str).eq("train")].copy()
    if len(train) < samples:
        raise ValueError(f"need at least {samples} train tensors for tiny overfit")
    # Pick tensors with some rain if possible, because an all-dry sample can pass
    # BCE while giving weak intensity-path evidence.
    train = train.sort_values(["target_missing_fraction", "event_id", "forecast_issue_time_utc"])
    records = [_load_tensor(Path(path)) for path in train["tensor_path"].head(samples)]
    batch = _prepare_inputs(records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    radar_input = batch["radar_input"].to(device)
    goes_input = batch["goes_input"].to(device)
    target_occurrence = batch["target_occurrence"].to(device)
    target_intensity = batch["target_intensity"].to(device)
    target_mask = batch["target_mask"].to(device)
    model = TinyRadarGoesFusion.build(
        radar_input.shape[1], goes_input.shape[1], target_occurrence.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    first_gradients: dict[str, float] | None = None
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        occurrence_logits, intensity_raw = model(radar_input, goes_input)
        loss, parts = radar_multitask_loss(
            occurrence_logits,
            intensity_raw,
            target_occurrence,
            target_intensity,
            target_mask,
            occurrence_weight=1.0,
            intensity_weight=0.5,
        )
        loss.backward()
        gradients = {
            "radar_branch": _grad_norm(model.radar_branch.parameters()),
            "satellite_branch": _grad_norm(model.satellite_branch.parameters()),
            "fusion": _grad_norm(model.fusion.parameters()),
        }
        if first_gradients is None:
            first_gradients = gradients
        optimizer.step()
        losses.append({"step": step, **parts, **{f"grad_{key}": value for key, value in gradients.items()}})
        if step == 1 or step % 10 == 0:
            print(f"[stage4-tiny-overfit] step {step}/{steps}: loss={parts['total_loss']:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        base_occ, base_intensity = model(radar_input, goes_input)
        zero_occ, zero_intensity = model(radar_input, torch.zeros_like(goes_input))
        shuffled_goes = goes_input.flip(0) if goes_input.shape[0] > 1 else torch.roll(goes_input, 1, 1)
        shuffled_occ, shuffled_intensity = model(radar_input, shuffled_goes)
        remove_satellite_delta = float(
            (
                torch.mean(torch.abs(base_occ - zero_occ))
                + torch.mean(torch.abs(base_intensity - zero_intensity))
            )
            .detach()
            .cpu()
        )
        shuffle_satellite_delta = float(
            (
                torch.mean(torch.abs(base_occ - shuffled_occ))
                + torch.mean(torch.abs(base_intensity - shuffled_intensity))
            )
            .detach()
            .cpu()
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(losses).to_csv(output_dir / "stage4_tiny_overfit_curve.csv", index=False)
    torch.save(model.state_dict(), output_dir / "stage4_tiny_overfit_model.pt")
    initial_loss = float(losses[0]["total_loss"])
    final_loss = float(losses[-1]["total_loss"])
    diagnostics = batch["diagnostics"]
    summary = {
        "created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "tensor_manifest": tensor_manifest.as_posix(),
        "samples": samples,
        "steps": steps,
        "device": str(device),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / max(initial_loss, 1e-9),
        "loss_decreased": final_loss < initial_loss,
        "first_gradient_norms": first_gradients,
        "radar_branch_receives_gradients": bool(first_gradients and first_gradients["radar_branch"] > 0),
        "satellite_branch_receives_gradients": bool(
            first_gradients and first_gradients["satellite_branch"] > 0
        ),
        "fusion_receives_gradients": bool(first_gradients and first_gradients["fusion"] > 0),
        "c13_values_nonconstant": bool(diagnostics["goes_raw_c13_std"] > 0),
        "cooling_channels_nonconstant": bool(diagnostics["goes_cooling_std"] > 0),
        "validity_masks_nonempty": bool(
            diagnostics["radar_mask_valid_fraction"] > 0
            and diagnostics["goes_mask_valid_fraction"] > 0
            and diagnostics["target_mask_valid_fraction"] > 0
        ),
        "removing_satellite_changes_output": bool(remove_satellite_delta > 1e-5),
        "shuffling_satellite_changes_output": bool(shuffle_satellite_delta > 1e-5),
        "remove_satellite_output_delta": remove_satellite_delta,
        "shuffle_satellite_output_delta": shuffle_satellite_delta,
        "input_diagnostics": diagnostics,
        "training_started": False,
    }
    checks = [
        summary["loss_decreased"],
        summary["radar_branch_receives_gradients"],
        summary["satellite_branch_receives_gradients"],
        summary["fusion_receives_gradients"],
        summary["c13_values_nonconstant"],
        summary["cooling_channels_nonconstant"],
        summary["validity_masks_nonempty"],
        summary["removing_satellite_changes_output"],
        summary["shuffling_satellite_changes_output"],
    ]
    summary["complete"] = bool(all(checks))
    (output_dir / "stage4_tiny_overfit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor-manifest",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4/tiny_overfit"))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    print(
        json.dumps(
            run_tiny_overfit(
                args.tensor_manifest,
                args.output_dir,
                samples=args.samples,
                steps=args.steps,
                learning_rate=args.learning_rate,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
