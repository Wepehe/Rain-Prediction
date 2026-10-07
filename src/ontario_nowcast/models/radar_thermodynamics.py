"""Capacity-controlled radar plus HRRR-thermodynamics model for Stage 4C."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .radar_convlstm import ConvBlock, ConvLSTMCell


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@dataclass(frozen=True)
class ThermodynamicParameterBreakdown:
    radar_pathway: int
    thermodynamic_branch: int
    fusion: int
    decoder_and_heads: int
    total: int


class RadarThermodynamicConvLSTM(nn.Module):
    def __init__(
        self,
        future_steps: int = 20,
        base_channels: int = 23,
        latent_channels: int = 72,
        thermo_base: int = 8,
        thermo_latent: int = 24,
        occurrence_thresholds: int = 4,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.occurrence_thresholds = occurrence_thresholds
        self.enc1 = ConvBlock(2, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = ConvBlock(base_channels * 2, latent_channels, stride=2)
        self.recurrent = ConvLSTMCell(latent_channels, latent_channels)
        self.thermo_enc1 = ConvBlock(10, thermo_base)
        self.thermo_enc2 = ConvBlock(thermo_base, thermo_base * 2, stride=2)
        self.thermo_enc3 = ConvBlock(thermo_base * 2, thermo_latent, stride=2)
        self.thermo_recurrent = ConvLSTMCell(thermo_latent, thermo_latent)
        self.thermo_to_radar = nn.Conv2d(thermo_latent, latent_channels, 1)
        nn.init.zeros_(self.thermo_to_radar.weight)
        nn.init.zeros_(self.thermo_to_radar.bias)
        self.dec2 = ConvBlock(latent_channels + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, future_steps * (occurrence_thresholds + 1), 1),
        )

    def forward(
        self, radar: torch.Tensor, thermo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rs = None
        s1 = s2 = None
        for step in range(radar.shape[1]):
            s1 = self.enc1(radar[:, step])
            s2 = self.enc2(s1)
            rs = self.recurrent(self.enc3(s2), rs)
        ts = None
        for step in range(thermo.shape[1]):
            t = self.thermo_enc1(thermo[:, step])
            t = self.thermo_enc2(t)
            ts = self.thermo_recurrent(self.thermo_enc3(t), ts)
        assert rs is not None and ts is not None and s1 is not None and s2 is not None
        hidden = rs[0] + self.thermo_to_radar(ts[0])
        d2 = self.dec2(
            torch.cat(
                [
                    F.interpolate(hidden, size=s2.shape[-2:], mode="bilinear", align_corners=False),
                    s2,
                ],
                1,
            )
        )
        d1 = self.dec1(
            torch.cat(
                [F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1
            )
        )
        out = self.head(d1)
        out = out.view(
            out.shape[0], self.future_steps, self.occurrence_thresholds + 1, *out.shape[-2:]
        )
        return out[:, :, : self.occurrence_thresholds], out[:, :, self.occurrence_thresholds]

    def load_radar_weights(self, state: dict[str, torch.Tensor]) -> int:
        own = self.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        self.load_state_dict({**own, **compatible})
        return len(compatible)

    def parameter_breakdown(self) -> ThermodynamicParameterBreakdown:
        radar = sum(_count(x) for x in (self.enc1, self.enc2, self.enc3, self.recurrent))
        thermo = sum(
            _count(x)
            for x in (self.thermo_enc1, self.thermo_enc2, self.thermo_enc3, self.thermo_recurrent)
        )
        fusion = _count(self.thermo_to_radar)
        decoder = sum(_count(x) for x in (self.dec2, self.dec1, self.head))
        return ThermodynamicParameterBreakdown(
            radar, thermo, fusion, decoder, radar + thermo + fusion + decoder
        )
