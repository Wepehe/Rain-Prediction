"""Optional network smoke: run manually with `uv run python tests/live_mrms_smoke.py`."""
from ontario_nowcast.operational import forecast, load_operational_model
from ontario_nowcast.operational.live_mrms import live_smoke_test

live = live_smoke_test()
result = forecast(load_operational_model(), live.history_rate, live.timestamps,
                  live.validity_mask, {"grid": {"x": live.tile.x.tolist(), "y": live.tile.y.tolist()}})
assert result.residual_probability.shape == (20, 128, 128)
print(f"live MRMS smoke passed at {live.issue_time.isoformat()}")
