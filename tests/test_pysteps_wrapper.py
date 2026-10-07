import sys
import types

import numpy as np

from ontario_nowcast.models.baselines import pysteps_deterministic_extrapolation


def test_pysteps_deterministic_wrapper_uses_public_interfaces(monkeypatch) -> None:
    history = np.ones((3, 4, 5), dtype=np.float32)
    calls = []

    def dB_transform(values, metadata=None, threshold=None, zerovalue=None, inverse=False):
        calls.append(("transform", inverse, threshold, zerovalue))
        if inverse:
            return values + 1.0, {}
        return values, {"zerovalue": -15.0}

    def motion_method(values):
        calls.append(("motion", values.shape))
        return np.zeros((2, values.shape[1], values.shape[2]), dtype=np.float32)

    def nowcast_method(latest, velocity, lead_steps):
        calls.append(("nowcast", latest.shape, velocity.shape, lead_steps))
        return np.repeat(latest[None, ...], lead_steps, axis=0)

    pysteps = types.ModuleType("pysteps")
    motion = types.SimpleNamespace(get_method=lambda name: motion_method)
    nowcasts = types.SimpleNamespace(get_method=lambda name: nowcast_method)
    utils = types.ModuleType("pysteps.utils")
    transformation = types.SimpleNamespace(dB_transform=dB_transform)
    pysteps.motion = motion
    pysteps.nowcasts = nowcasts
    utils.transformation = transformation
    monkeypatch.setitem(sys.modules, "pysteps", pysteps)
    monkeypatch.setitem(sys.modules, "pysteps.utils", utils)
    monkeypatch.setitem(sys.modules, "pysteps.utils.transformation", transformation)

    forecast = pysteps_deterministic_extrapolation(history, 2)

    assert forecast.shape == (2, 4, 5)
    assert np.all(forecast == 2.0)
    assert calls[1][0] == "motion"
    assert calls[2][0] == "nowcast"
