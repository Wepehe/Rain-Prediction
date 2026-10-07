import torch

from ontario_nowcast.models.radar_goes import RadarGoesConvLSTM


def test_stage4b_radar_goes_model_outputs_multithreshold_heads() -> None:
    model = RadarGoesConvLSTM(
        radar_input_channels=2,
        goes_input_channels=10,
        future_steps=3,
        base_channels=4,
        latent_channels=8,
        satellite_base_channels=4,
        satellite_latent_channels=6,
        occurrence_thresholds=4,
    )

    occurrence_logits, intensity_raw = model(
        torch.zeros(2, 4, 2, 32, 32),
        torch.zeros(2, 3, 10, 32, 32),
    )

    assert occurrence_logits.shape == (2, 3, 4, 32, 32)
    assert intensity_raw.shape == (2, 3, 32, 32)


def test_stage4b_parameter_breakdown_sums_to_total() -> None:
    model = RadarGoesConvLSTM(
        radar_input_channels=2,
        goes_input_channels=2,
        future_steps=3,
        base_channels=4,
        latent_channels=8,
        satellite_base_channels=4,
        satellite_latent_channels=6,
        occurrence_thresholds=4,
    )

    breakdown = model.parameter_breakdown()

    assert breakdown.total > breakdown.radar_branch
    assert (
        breakdown.radar_branch
        + breakdown.satellite_branch
        + breakdown.fusion_layers
        + breakdown.decoder_and_heads
        == breakdown.total
    )
