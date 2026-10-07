"""Radar-only ConvLSTM nowcasting models for Stage 3 experiments."""

from __future__ import annotations

from math import gcd
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell with same-padding convolutions."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self, inputs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            batch, _, height, width = inputs.shape
            hidden = inputs.new_zeros(batch, self.hidden_channels, height, width)
            cell = inputs.new_zeros(batch, self.hidden_channels, height, width)
        else:
            hidden, cell = state
        ingate, forgetgate, outgate, candidate = self.gates(
            torch.cat([inputs, hidden], dim=1)
        ).chunk(4, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        outgate = torch.sigmoid(outgate)
        candidate = torch.tanh(candidate)
        cell = forgetgate * cell + ingate * candidate
        hidden = outgate * torch.tanh(cell)
        return hidden, cell


class RadarConvLSTM(nn.Module):
    """Modest multi-head radar-evolution model.

    Input shape is ``(batch, history, channels, y, x)``. Output tensors have shape
    ``(batch, future, y, x)`` for both occurrence logits and conditional log-intensity.
    """

    def __init__(
        self,
        *,
        input_channels: int = 2,
        future_steps: int = 20,
        encoder_channels: int = 16,
        hidden_channels: int = 16,
        decoder_channels: int = 16,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.recurrent = ConvLSTMCell(encoder_channels, hidden_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, future_steps * 2, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 5:
            raise ValueError("inputs must have shape (batch, time, channel, y, x)")
        state = None
        for step in range(inputs.shape[1]):
            encoded = self.encoder(inputs[:, step])
            state = self.recurrent(encoded, state)
        assert state is not None
        decoded = self.decoder(state[0])
        occurrence_logits, intensity_raw = decoded.chunk(2, dim=1)
        return occurrence_logits, intensity_raw


class ConvBlock(nn.Module):
    """Small convolutional block used by the Stage 3.1 multi-scale model."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = gcd(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class MultiScaleRadarConvLSTM(nn.Module):
    """Modest radar-only encoder/ConvLSTM/decoder control for Stage 3.1.

    The occurrence head can predict one or more exceedance probabilities. With the
    default Stage 3.1 config these channels are P(rate > 0.1), P(rate > 1),
    P(rate > 2.5), and P(rate > 5 mm/h). The intensity head remains a conditional
    log-intensity field with shape ``(batch, future, y, x)``.
    """

    def __init__(
        self,
        *,
        input_channels: int = 2,
        future_steps: int = 20,
        base_channels: int = 32,
        latent_channels: int = 96,
        occurrence_thresholds: int = 4,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.occurrence_thresholds = occurrence_thresholds
        c1 = base_channels
        c2 = base_channels * 2
        c3 = latent_channels
        self.enc1 = ConvBlock(input_channels, c1)
        self.enc2 = ConvBlock(c1, c2, stride=2)
        self.enc3 = ConvBlock(c2, c3, stride=2)
        self.recurrent = ConvLSTMCell(c3, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, future_steps * (occurrence_thresholds + 1), kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 5:
            raise ValueError("inputs must have shape (batch, time, channel, y, x)")
        state = None
        last_skip1 = None
        last_skip2 = None
        for step in range(inputs.shape[1]):
            skip1 = self.enc1(inputs[:, step])
            skip2 = self.enc2(skip1)
            encoded = self.enc3(skip2)
            state = self.recurrent(encoded, state)
            last_skip1 = skip1
            last_skip2 = skip2
        assert state is not None and last_skip1 is not None and last_skip2 is not None
        hidden = state[0]
        up2 = F.interpolate(hidden, size=last_skip2.shape[-2:], mode="bilinear", align_corners=False)
        decoded2 = self.dec2(torch.cat([up2, last_skip2], dim=1))
        up1 = F.interpolate(decoded2, size=last_skip1.shape[-2:], mode="bilinear", align_corners=False)
        decoded1 = self.dec1(torch.cat([up1, last_skip1], dim=1))
        output = self.head(decoded1)
        batch, _, height, width = output.shape
        output = output.view(
            batch,
            self.future_steps,
            self.occurrence_thresholds + 1,
            height,
            width,
        )
        occurrence_logits = output[:, :, : self.occurrence_thresholds]
        intensity_raw = output[:, :, self.occurrence_thresholds]
        return occurrence_logits, intensity_raw


def build_radar_model(config: dict[str, Any]) -> nn.Module:
    """Build the radar-only model requested by an experiment config."""
    model_cfg = config["model"]
    data_cfg = config["data"]
    architecture = model_cfg.get("architecture", "minimal_convlstm")
    if architecture in {"minimal_convlstm", "radar_convlstm", "compact_convlstm_encoder_decoder"}:
        return RadarConvLSTM(
            input_channels=int(model_cfg["input_channels"]),
            future_steps=int(data_cfg["target_frames"]),
            encoder_channels=int(model_cfg["encoder_channels"]),
            hidden_channels=int(model_cfg["hidden_channels"]),
            decoder_channels=int(model_cfg["decoder_channels"]),
        )
    if architecture == "multiscale_convlstm_encoder_decoder":
        return MultiScaleRadarConvLSTM(
            input_channels=int(model_cfg["input_channels"]),
            future_steps=int(data_cfg["target_frames"]),
            base_channels=int(model_cfg["base_channels"]),
            latent_channels=int(model_cfg["latent_channels"]),
            occurrence_thresholds=len(model_cfg.get("occurrence_thresholds_mm_hr", [0.1])),
        )
    raise ValueError(f"unknown radar model architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def expected_rate_mm_hr(
    occurrence_logits: torch.Tensor, intensity_raw: torch.Tensor
) -> torch.Tensor:
    """Convert model heads into expected rain rate in mm/h."""
    occurrence_probability = torch.sigmoid(occurrence_logits)
    if occurrence_probability.ndim == 5:
        occurrence_probability = occurrence_probability[:, :, 0]
    conditional_log_intensity = torch.nn.functional.softplus(intensity_raw)
    conditional_rate = torch.expm1(conditional_log_intensity)
    return occurrence_probability * conditional_rate
