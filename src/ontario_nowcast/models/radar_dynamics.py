"""A+-controlled radar model with a compact HRRR-dynamics residual branch."""

from __future__ import annotations

from dataclasses import dataclass

from .radar_convlstm import ConvBlock
from .radar_thermodynamics import RadarThermodynamicConvLSTM, _count


@dataclass(frozen=True)
class DynamicsParameterBreakdown:
    radar_pathway: int
    dynamics_branch: int
    fusion: int
    decoder_and_heads: int
    total: int


class RadarDynamicsConvLSTM(RadarThermodynamicConvLSTM):
    """Same radar/decoder control as C1, with 10 fields plus 10 masks."""

    def __init__(self) -> None:
        super().__init__()
        self.thermo_enc1 = ConvBlock(20, 8)

    def parameter_breakdown(self) -> DynamicsParameterBreakdown:
        base = super().parameter_breakdown()
        return DynamicsParameterBreakdown(
            radar_pathway=base.radar_pathway,
            dynamics_branch=base.thermodynamic_branch,
            fusion=base.fusion,
            decoder_and_heads=base.decoder_and_heads,
            total=sum(_count(module) for module in self.children()),
        )
