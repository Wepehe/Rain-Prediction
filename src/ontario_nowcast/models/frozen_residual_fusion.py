"""Small GOES+HRRR correction network over immutable A+ output fields."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .radar_convlstm import ConvBlock, ConvLSTMCell


class FrozenBaselineResidualFusion(nn.Module):
    def __init__(self, future_steps: int = 20) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.goes1 = ConvBlock(10, 8)
        self.goes2 = ConvBlock(8, 16, stride=2)
        self.goes3 = ConvBlock(16, 24, stride=2)
        self.goes_recurrent = ConvLSTMCell(24, 24)
        self.context = nn.Sequential(ConvBlock(8, 16, stride=2), ConvBlock(16, 24, stride=2))
        self.fusion = nn.Sequential(ConvBlock(48, 32), nn.Conv2d(32, 5, 1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, base_logits, base_intensity_raw, base_expected_rate, hrrr, goes):
        state = None
        for step in range(goes.shape[1]):
            state = self.goes_recurrent(self.goes3(self.goes2(self.goes1(goes[:, step]))), state)
        assert state is not None
        goes_latent = state[0]
        outputs = []
        for step in range(self.future_steps):
            hour = hrrr[:, 0 if step < 10 else 1]
            context = torch.cat(
                [
                    base_logits[:, step],
                    base_intensity_raw[:, step, None],
                    torch.log1p(base_expected_rate[:, step, None].clamp_min(0)),
                    hour,
                ],
                1,
            )
            delta = self.fusion(torch.cat([self.context(context), goes_latent], 1))
            outputs.append(
                F.interpolate(
                    delta, size=base_logits.shape[-2:], mode="bilinear", align_corners=False
                )
            )
        raw = torch.stack(outputs, 1)
        final_logits = base_logits + 4.0 * torch.tanh(raw[:, :, :4])
        final_intensity = base_intensity_raw + 2.0 * torch.tanh(raw[:, :, 4])
        return final_logits, final_intensity, raw


class NonRadarPredictabilityProbe(nn.Module):
    """Compact HRRR+GOES occurrence probe with no radar inputs or parameters."""

    def __init__(self, future_steps: int = 20) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.goes1 = ConvBlock(10, 8)
        self.goes2 = ConvBlock(8, 16, stride=2)
        self.goes3 = ConvBlock(16, 24, stride=2)
        self.goes_recurrent = ConvLSTMCell(24, 24)
        self.hrrr = nn.Sequential(ConvBlock(2, 12, stride=2), ConvBlock(12, 16, stride=2))
        self.head = nn.Sequential(ConvBlock(40, 24), nn.Conv2d(24, 1, 1))

    def forward(self, hrrr, goes):
        state = None
        for step in range(goes.shape[1]):
            state = self.goes_recurrent(self.goes3(self.goes2(self.goes1(goes[:, step]))), state)
        assert state is not None
        outputs = []
        for step in range(self.future_steps):
            hour = hrrr[:, 0 if step < 10 else 1]
            logits = self.head(torch.cat([self.hrrr(hour), state[0]], 1))
            outputs.append(
                F.interpolate(logits, size=hrrr.shape[-2:], mode="bilinear", align_corners=False)
            )
        return torch.stack(outputs, 1)[:, :, 0]
