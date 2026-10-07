"""Compact Cycle-2 residual U-Net built around deterministic PySTEPS."""
from __future__ import annotations

from math import gcd
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, inputs: int, outputs: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = gcd(8, outputs)
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1),
            nn.GroupNorm(groups, outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.GroupNorm(groups, outputs), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PySTEPSResidualUNetV1(nn.Module):
    """Treat history, PySTEPS future, motion, and validity as 2-D channels.

    ``radar_history`` is Bx10xHxW, ``pysteps_rate`` is Bx20xHxW,
    ``motion`` is Bx2xHxW, and an optional validity tensor is Bx10xHxW.
    The intensity head is exactly identity at initialization.
    """

    def __init__(self, *, include_validity: bool = True, widths=(40, 80, 160, 320)) -> None:
        super().__init__()
        self.include_validity = include_validity
        self.register_buffer("rate_mean", torch.tensor(0.0))
        self.register_buffer("rate_std", torch.tensor(1.0))
        self.register_buffer("motion_mean", torch.zeros(2).view(1, 2, 1, 1))
        self.register_buffer("motion_std", torch.ones(2).view(1, 2, 1, 1))
        inputs = 10 + 20 + 2 + (10 if include_validity else 0)
        w1, w2, w3, w4 = widths
        self.enc1 = DoubleConv(inputs, w1)
        self.enc2 = DoubleConv(w1, w2, stride=2)
        self.enc3 = DoubleConv(w2, w3, stride=2)
        self.enc4 = DoubleConv(w3, w4, stride=2)
        self.dec3 = DoubleConv(w4 + w3, w3)
        self.dec2 = DoubleConv(w3 + w2, w2)
        self.dec1 = DoubleConv(w2 + w1, w1)
        self.occurrence_head = nn.Conv2d(w1, 20, 1)
        self.residual_head = nn.Conv2d(w1, 20, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def set_normalization(self, *, rate_mean, rate_std, motion_mean, motion_std) -> None:
        if float(rate_std) <= 0 or torch.as_tensor(motion_std).min().item() <= 0:
            raise ValueError("normalization standard deviations must be positive")
        self.rate_mean.fill_(float(rate_mean)); self.rate_std.fill_(float(rate_std))
        self.motion_mean.copy_(torch.as_tensor(motion_mean).view(1, 2, 1, 1))
        self.motion_std.copy_(torch.as_tensor(motion_std).view(1, 2, 1, 1))

    def forward(self, radar_history, pysteps_rate, motion, validity=None):
        history_log = torch.log1p(torch.clamp(radar_history, min=0))
        baseline_log = torch.log1p(torch.clamp(pysteps_rate, min=0))
        tensors = [(history_log-self.rate_mean)/self.rate_std,
                   (baseline_log-self.rate_mean)/self.rate_std,
                   (motion-self.motion_mean)/self.motion_std]
        if self.include_validity:
            if validity is None:
                raise ValueError("validity is required when include_validity=True")
            tensors.append(validity.float())
        x1 = self.enc1(torch.cat(tensors, dim=1))
        x2 = self.enc2(x1); x3 = self.enc3(x2); x4 = self.enc4(x3)
        y3 = self.dec3(torch.cat([F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False), x3], 1))
        y2 = self.dec2(torch.cat([F.interpolate(y3, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2], 1))
        y1 = self.dec1(torch.cat([F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1], 1))
        logits = self.occurrence_head(y1)
        delta = self.residual_head(y1)
        corrected_log_rate = baseline_log + delta
        corrected_rate = torch.expm1(torch.clamp(corrected_log_rate, min=0))
        return {"occurrence_logits": logits, "delta_log_rate": delta,
                "corrected_log_rate": corrected_log_rate, "corrected_rate": corrected_rate}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
