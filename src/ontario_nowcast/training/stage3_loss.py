"""Losses for the Stage 3 radar-only experiment."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def occurrence_targets_from_intensity(
    target_intensity: torch.Tensor,
    target_mask: torch.Tensor,
    thresholds_mm_hr: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return multi-threshold occurrence targets and matching validity mask."""
    target_rate = torch.expm1(target_intensity)
    thresholds = target_rate.new_tensor(thresholds_mm_hr).view(
        *([1] * target_rate.ndim), len(thresholds_mm_hr)
    )
    targets = (target_rate.unsqueeze(-1) >= thresholds).float()
    mask = target_mask.unsqueeze(-1).expand_as(targets)
    return targets, mask


def radar_multitask_loss(
    occurrence_logits: torch.Tensor,
    intensity_raw: torch.Tensor,
    target_occurrence: torch.Tensor,
    target_intensity: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    occurrence_weight: float = 1.0,
    intensity_weight: float = 0.5,
    occurrence_thresholds_mm_hr: list[float] | None = None,
    occurrence_pos_weight: torch.Tensor | list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """BCE occurrence plus rainy-pixel log-intensity Smooth L1.

    Missing target pixels are masked out. The intensity term is evaluated only where observed
    precipitation exceeds the occurrence threshold.
    """
    if occurrence_logits.ndim == target_occurrence.ndim + 1:
        thresholds = occurrence_thresholds_mm_hr or [0.1] * occurrence_logits.shape[2]
        occurrence_target, occurrence_mask = occurrence_targets_from_intensity(
            target_intensity, target_mask, [float(value) for value in thresholds]
        )
        occurrence_target = occurrence_target.movedim(-1, 2)
        occurrence_mask = occurrence_mask.movedim(-1, 2) > 0
    else:
        occurrence_target = target_occurrence
        occurrence_mask = target_mask > 0

    valid = target_mask > 0
    if not torch.any(occurrence_mask):
        zero = occurrence_logits.sum() * 0.0
        return zero, {"occurrence_loss": 0.0, "intensity_loss": 0.0, "total_loss": 0.0}
    if occurrence_pos_weight is None:
        occurrence_loss = F.binary_cross_entropy_with_logits(
            occurrence_logits[occurrence_mask], occurrence_target[occurrence_mask]
        )
    else:
        pos_weight = torch.as_tensor(
            occurrence_pos_weight, dtype=occurrence_logits.dtype, device=occurrence_logits.device
        )
        if occurrence_logits.ndim == target_occurrence.ndim + 1:
            weights = pos_weight.view(1, 1, -1, 1, 1)
        else:
            weights = pos_weight.reshape(())
        raw_loss = F.binary_cross_entropy_with_logits(
            occurrence_logits,
            occurrence_target,
            pos_weight=weights,
            reduction="none",
        )
        occurrence_loss = raw_loss[occurrence_mask].mean()
    rainy = valid & (target_occurrence > 0)
    if torch.any(rainy):
        predicted_log_intensity = F.softplus(intensity_raw[rainy])
        intensity_loss = F.smooth_l1_loss(predicted_log_intensity, target_intensity[rainy])
    else:
        intensity_loss = occurrence_logits.sum() * 0.0
    total = occurrence_weight * occurrence_loss + intensity_weight * intensity_loss
    return total, {
        "occurrence_loss": float(occurrence_loss.detach().cpu()),
        "intensity_loss": float(intensity_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


def loss_kwargs_from_config(loss_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build keyword arguments for ``radar_multitask_loss`` from config."""
    kwargs: dict[str, Any] = {
        "occurrence_weight": float(loss_cfg["occurrence_weight"]),
        "intensity_weight": float(loss_cfg["intensity_weight"]),
        "occurrence_thresholds_mm_hr": [
            float(value) for value in model_cfg.get("occurrence_thresholds_mm_hr", [0.1])
        ],
    }
    if "occurrence_pos_weight" in loss_cfg:
        kwargs["occurrence_pos_weight"] = loss_cfg["occurrence_pos_weight"]
    return kwargs
