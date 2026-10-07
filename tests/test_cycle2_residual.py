import torch

from ontario_nowcast.models.pysteps_residual_unet import PySTEPSResidualUNetV1, count_parameters
from ontario_nowcast.training.cycle2_residual import EventBalancedBatchPlan


def test_residual_model_shapes_probability_and_identity_initialization():
    model = PySTEPSResidualUNetV1()
    history = torch.rand(1, 10, 32, 32)
    baseline = torch.rand(1, 20, 32, 32)
    output = model(history, baseline, torch.zeros(1, 2, 32, 32), torch.ones_like(history))
    assert output["occurrence_logits"].shape == (1, 20, 32, 32)
    assert torch.equal(output["delta_log_rate"], torch.zeros_like(output["delta_log_rate"]))
    assert torch.allclose(output["corrected_rate"], baseline, atol=1e-6)
    probability = torch.sigmoid(output["occurrence_logits"])
    assert bool(((probability >= 0) & (probability <= 1)).all())
    assert 3_000_000 <= count_parameters(model) <= 8_000_000


def test_validity_is_explicitly_required():
    model = PySTEPSResidualUNetV1(include_validity=True)
    try:
        model(torch.zeros(1, 10, 16, 16), torch.zeros(1, 20, 16, 16), torch.zeros(1, 2, 16, 16))
    except ValueError as exc:
        assert "validity" in str(exc)
    else:
        raise AssertionError("missing validity mask was accepted")


def test_batch_plan_limits_negative_injection_to_one_per_batch():
    import pandas as pd
    rows = pd.DataFrame([
        {"system_id": "a", "row_type": "active_precip"},
        {"system_id": "a", "row_type": "clean_initiation"},
        {"system_id": "b", "row_type": "active_precip"},
        {"system_id": "b", "row_type": "hard_negative"},
        {"system_id": "c", "row_type": "hard_negative"},
    ])
    batches, audit = EventBalancedBatchPlan(rows, batch_size=3, batches_per_epoch=8).epoch(1)
    negative = {3, 4}
    assert all(sum(i in negative for i in batch) <= 1 for batch in batches)
    assert audit["hard_negative_draws"] == 2
