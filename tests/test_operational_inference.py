import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from ontario_nowcast.operational import inference


class FakeModel(torch.nn.Module):
    def forward(self, history, baseline, motion, validity):
        return {
            "occurrence_logits": torch.zeros_like(baseline),
            "corrected_rate": baseline,
        }


def operational(monkeypatch):
    monkeypatch.setattr(
        inference,
        "deterministic_pysteps_with_motion",
        lambda history: (
            np.broadcast_to(history[-1], (20, 128, 128)).copy(),
            np.zeros((2, 128, 128), np.float32),
            "none",
        ),
    )
    manifest = json.loads(Path("artifacts/operational/residual_v1/manifest.json").read_text())
    return inference.OperationalModel(FakeModel(), torch.device("cpu"), manifest, Path("."))


def valid_input():
    end = datetime(2025, 1, 1, tzinfo=UTC)
    times = [end + timedelta(minutes=6 * i) for i in range(10)]
    return np.ones((10, 128, 128), np.float32), times, np.ones((10, 128, 128), bool)


def test_shapes_threshold_and_reproducibility(monkeypatch):
    model = operational(monkeypatch)
    history, times, valid = valid_input()
    first = inference.forecast(model, history, times, valid)
    second = inference.forecast(model, history, times, valid)
    assert first.pysteps_rate_mm_h.shape == (20, 128, 128)
    assert first.residual_probability.shape == (20, 128, 128)
    assert first.residual_rate_mm_h.shape == (20, 128, 128)
    assert np.all((first.residual_probability >= 0) & (first.residual_probability <= 1))
    assert np.all(first.residual_wet_mask)  # sigmoid(0)=0.5 >= frozen 0.35
    assert np.array_equal(first.residual_probability, second.residual_probability)
    assert first.metadata["operating_threshold"] == 0.35


@pytest.mark.parametrize("shape", [(9, 128, 128), (11, 128, 128), (10, 64, 64)])
def test_wrong_history_shape_rejected(monkeypatch, shape):
    model = operational(monkeypatch)
    _, times, _ = valid_input()
    with pytest.raises(ValueError, match="shape"):
        inference.forecast(model, np.ones(shape), times, np.ones(shape, bool))


def test_bad_cadence_order_and_all_invalid_rejected(monkeypatch):
    model = operational(monkeypatch)
    history, times, valid = valid_input()
    bad = list(times); bad[-1] += timedelta(minutes=1)
    with pytest.raises(ValueError, match="six-minute"):
        inference.forecast(model, history, bad, valid)
    reverse = list(reversed(times))
    with pytest.raises(ValueError, match="increasing"):
        inference.forecast(model, history, reverse, valid)
    with pytest.raises(ValueError, match="entirely invalid"):
        inference.forecast(model, history, times, np.zeros_like(valid))


def test_missing_values_are_masked(monkeypatch):
    model = operational(monkeypatch)
    history, times, valid = valid_input()
    history[0, 0, 0] = np.nan
    result = inference.forecast(model, history, times, valid)
    assert not result.validity_mask[0, 0, 0]


def test_bundle_hashes_normalization_and_checkpoint_load():
    model = inference.load_operational_model()
    assert model.model.rate_mean.item() == pytest.approx(0.27863315186335164)
    assert sum(p.numel() for p in model.model.parameters()) == 3_060_440


def test_hash_mismatch_fails_loudly(tmp_path):
    bundle = Path("artifacts/operational/residual_v1")
    for name in ("manifest.json", "checkpoint.pt", "normalization.json", "config.yaml", "model_source.py"):
        (tmp_path / name).write_bytes((bundle / name).read_bytes())
    (tmp_path / "normalization.json").write_text("{}")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        inference.load_operational_model(tmp_path)


def test_golden_fixture_is_dev_only_and_has_stable_hash():
    root = Path("tests/fixtures/operational_residual_v1")
    golden = json.loads((root / "golden.json").read_text())
    assert golden["split"] == "new_dev"
    assert "final" not in golden["row_id"].lower()
    assert hashlib.sha256((root / "dev_input.npz").read_bytes()).hexdigest() == golden["input_sha256"]


def test_operational_module_has_no_manifest_dependency():
    source = Path(inference.__file__).read_text().lower()
    assert "train_dev_rows" not in source
    assert "final_manifest" not in source
