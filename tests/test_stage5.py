import torch

from ontario_nowcast.models.frozen_residual_fusion import FrozenBaselineResidualFusion


def test_zero_initialized_h1_is_exactly_frozen_baseline() -> None:
    model = FrozenBaselineResidualFusion(future_steps=2)
    logits = torch.randn(1, 2, 4, 16, 16)
    intensity = torch.randn(1, 2, 16, 16)
    expected = torch.rand(1, 2, 16, 16)
    hrrr = torch.randn(1, 2, 2, 16, 16)
    goes = torch.randn(1, 3, 10, 16, 16)
    final_logits, final_intensity, residual = model(logits, intensity, expected, hrrr, goes)
    assert torch.equal(final_logits, logits)
    assert torch.equal(final_intensity, intensity)
    assert torch.count_nonzero(residual) == 0
