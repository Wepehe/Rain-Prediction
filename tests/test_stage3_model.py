import torch

from ontario_nowcast.models.radar_convlstm import (
    MultiScaleRadarConvLSTM,
    RadarConvLSTM,
    count_parameters,
    expected_rate_mm_hr,
)
from ontario_nowcast.training.stage3_loss import radar_multitask_loss


def test_radar_convlstm_outputs_future_multihead_fields() -> None:
    model = RadarConvLSTM(
        input_channels=2,
        future_steps=20,
        encoder_channels=4,
        hidden_channels=4,
        decoder_channels=4,
    )
    occurrence_logits, intensity_raw = model(torch.zeros(2, 10, 2, 16, 16))
    assert occurrence_logits.shape == (2, 20, 16, 16)
    assert intensity_raw.shape == (2, 20, 16, 16)
    expected = expected_rate_mm_hr(occurrence_logits, intensity_raw)
    assert torch.all(expected >= 0)


def test_stage3_loss_masks_missing_targets() -> None:
    occurrence_logits = torch.zeros(1, 2, 4, 4, requires_grad=True)
    intensity_raw = torch.zeros(1, 2, 4, 4, requires_grad=True)
    target_occurrence = torch.zeros(1, 2, 4, 4)
    target_intensity = torch.zeros(1, 2, 4, 4)
    target_mask = torch.zeros(1, 2, 4, 4)
    loss, parts = radar_multitask_loss(
        occurrence_logits,
        intensity_raw,
        target_occurrence,
        target_intensity,
        target_mask,
    )
    assert loss.item() == 0.0
    assert parts["total_loss"] == 0.0


def test_multiscale_radar_convlstm_outputs_multi_threshold_occurrence() -> None:
    model = MultiScaleRadarConvLSTM(
        input_channels=2,
        future_steps=3,
        base_channels=8,
        latent_channels=16,
        occurrence_thresholds=4,
    )
    occurrence_logits, intensity_raw = model(torch.zeros(2, 4, 2, 32, 32))
    assert occurrence_logits.shape == (2, 3, 4, 32, 32)
    assert intensity_raw.shape == (2, 3, 32, 32)
    assert expected_rate_mm_hr(occurrence_logits, intensity_raw).shape == (2, 3, 32, 32)
    assert count_parameters(model) > count_parameters(
        RadarConvLSTM(
            input_channels=2,
            future_steps=3,
            encoder_channels=4,
            hidden_channels=4,
            decoder_channels=4,
        )
    )
