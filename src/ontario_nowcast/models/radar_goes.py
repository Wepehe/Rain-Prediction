"""Radar + GOES C13 models for Stage 4B ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .radar_convlstm import ConvBlock, ConvLSTMCell


@dataclass(frozen=True)
class ParameterBreakdown:
    """Trainable-parameter counts for Stage 4B reporting."""

    total: int
    radar_branch: int
    satellite_branch: int
    fusion_layers: int
    decoder_and_heads: int


def _count_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


class RadarGoesConvLSTM(nn.Module):
    """Stage 4B radar+GOES model built around the frozen Stage 3.1 architecture.

    The radar pathway mirrors ``MultiScaleRadarConvLSTM``: multi-scale radar
    encoder, ConvLSTM latent recurrence, skip-connected decoder, exceedance
    occurrence heads, and conditional log-intensity head. GOES C13 inputs are
    encoded through a small temporal branch and fused into the radar latent state
    as an additive residual.
    """

    def __init__(
        self,
        *,
        radar_input_channels: int = 2,
        goes_input_channels: int = 2,
        future_steps: int = 20,
        base_channels: int = 18,
        latent_channels: int = 72,
        satellite_base_channels: int = 8,
        satellite_latent_channels: int = 24,
        occurrence_thresholds: int = 4,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.occurrence_thresholds = occurrence_thresholds
        c1 = base_channels
        c2 = base_channels * 2
        c3 = latent_channels
        s1 = satellite_base_channels
        s2 = satellite_base_channels * 2
        s3 = satellite_latent_channels

        self.enc1 = ConvBlock(radar_input_channels, c1)
        self.enc2 = ConvBlock(c1, c2, stride=2)
        self.enc3 = ConvBlock(c2, c3, stride=2)
        self.recurrent = ConvLSTMCell(c3, c3)

        self.sat_enc1 = ConvBlock(goes_input_channels, s1)
        self.sat_enc2 = ConvBlock(s1, s2, stride=2)
        self.sat_enc3 = ConvBlock(s2, s3, stride=2)
        self.sat_recurrent = ConvLSTMCell(s3, s3)
        self.sat_to_radar_latent = nn.Conv2d(s3, c3, kernel_size=1)
        # Start as close as possible to the frozen radar-only control. The
        # satellite branch initially contributes zero and learns a residual.
        nn.init.zeros_(self.sat_to_radar_latent.weight)
        nn.init.zeros_(self.sat_to_radar_latent.bias)

        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, future_steps * (occurrence_thresholds + 1), kernel_size=1),
        )

    def forward(
        self,
        radar_inputs: torch.Tensor,
        goes_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if radar_inputs.ndim != 5:
            raise ValueError("radar_inputs must have shape (batch, time, channel, y, x)")
        if goes_inputs.ndim != 5:
            raise ValueError("goes_inputs must have shape (batch, time, channel, y, x)")
        radar_state = None
        last_skip1 = None
        last_skip2 = None
        for step in range(radar_inputs.shape[1]):
            skip1 = self.enc1(radar_inputs[:, step])
            skip2 = self.enc2(skip1)
            encoded = self.enc3(skip2)
            radar_state = self.recurrent(encoded, radar_state)
            last_skip1 = skip1
            last_skip2 = skip2
        satellite_state = None
        for step in range(goes_inputs.shape[1]):
            sat1 = self.sat_enc1(goes_inputs[:, step])
            sat2 = self.sat_enc2(sat1)
            sat3 = self.sat_enc3(sat2)
            satellite_state = self.sat_recurrent(sat3, satellite_state)
        assert (
            radar_state is not None
            and satellite_state is not None
            and last_skip1 is not None
            and last_skip2 is not None
        )
        hidden = radar_state[0] + self.sat_to_radar_latent(satellite_state[0])
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

    def load_radar_control_weights(self, state_dict: dict[str, Any]) -> list[str]:
        """Load matching Stage 3.1 radar/decoder/head weights without touching GOES."""
        own_state = self.state_dict()
        loaded = []
        compatible = {}
        for key, value in state_dict.items():
            if key in own_state and tuple(own_state[key].shape) == tuple(value.shape):
                compatible[key] = value
                loaded.append(key)
        self.load_state_dict({**own_state, **compatible})
        return loaded

    def parameter_breakdown(self) -> ParameterBreakdown:
        radar = (
            _count_parameters(self.enc1)
            + _count_parameters(self.enc2)
            + _count_parameters(self.enc3)
            + _count_parameters(self.recurrent)
        )
        satellite = (
            _count_parameters(self.sat_enc1)
            + _count_parameters(self.sat_enc2)
            + _count_parameters(self.sat_enc3)
            + _count_parameters(self.sat_recurrent)
        )
        fusion = _count_parameters(self.sat_to_radar_latent)
        decoder = _count_parameters(self.dec2) + _count_parameters(self.dec1) + _count_parameters(self.head)
        return ParameterBreakdown(
            total=radar + satellite + fusion + decoder,
            radar_branch=radar,
            satellite_branch=satellite,
            fusion_layers=fusion,
            decoder_and_heads=decoder,
        )


def build_stage4b_model(config: dict[str, Any]) -> RadarGoesConvLSTM:
    """Build the Stage 4B radar+GOES model requested by a config."""
    model_cfg = config["model"]
    data_cfg = config["data"]
    return RadarGoesConvLSTM(
        radar_input_channels=int(model_cfg.get("radar_input_channels", 2)),
        goes_input_channels=int(model_cfg["goes_input_channels"]),
        future_steps=int(data_cfg.get("target_frames", 20)),
        base_channels=int(model_cfg.get("base_channels", 18)),
        latent_channels=int(model_cfg.get("latent_channels", 72)),
        satellite_base_channels=int(model_cfg.get("satellite_base_channels", 8)),
        satellite_latent_channels=int(model_cfg.get("satellite_latent_channels", 24)),
        occurrence_thresholds=len(model_cfg.get("occurrence_thresholds_mm_hr", [0.1])),
    )
