"""Persistence and deterministic optical-flow baselines."""

from __future__ import annotations

import contextlib
import io

import numpy as np
from scipy.ndimage import map_coordinates


def persistence(latest: np.ndarray, lead_steps: int) -> np.ndarray:
    if lead_steps < 1:
        raise ValueError("lead_steps must be positive")
    frame = np.asarray(latest, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError("latest must be two-dimensional")
    return np.repeat(frame[None, ...], lead_steps, axis=0)


def _dense_flow(previous: np.ndarray, latest: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - installation-specific
        raise RuntimeError("Install the 'baseline' extra to run optical flow") from exc

    def normalize(frame: np.ndarray) -> np.ndarray:
        finite = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = np.log1p(np.clip(finite, 0.0, None))
        high = np.percentile(scaled, 99.5)
        return np.uint8(np.clip(scaled / max(high, 1e-6) * 255.0, 0, 255))

    return cv2.calcOpticalFlowFarneback(
        normalize(previous), normalize(latest), None, 0.5, 4, 25, 4, 7, 1.5, 0
    )


def optical_flow_extrapolation(
    previous: np.ndarray, latest: np.ndarray, lead_steps: int
) -> np.ndarray:
    """Estimate one-step flow and advect the latest rate field with semi-Lagrangian sampling."""
    if lead_steps < 1:
        raise ValueError("lead_steps must be positive")
    previous = np.asarray(previous, dtype=np.float32)
    latest = np.asarray(latest, dtype=np.float32)
    if previous.shape != latest.shape or latest.ndim != 2:
        raise ValueError("previous and latest must be same-shaped two-dimensional arrays")
    flow = _dense_flow(previous, latest)
    y, x = np.indices(latest.shape, dtype=np.float32)
    forecasts = []
    for step in range(1, lead_steps + 1):
        source_y = y - step * flow[..., 1]
        source_x = x - step * flow[..., 0]
        forecast = map_coordinates(
            np.nan_to_num(latest, nan=0.0),
            [source_y, source_x],
            order=1,
            mode="constant",
            cval=0.0,
        )
        forecasts.append(np.clip(forecast, 0.0, None))
    return np.stack(forecasts)


def _pysteps_modules():
    try:
        from pysteps import motion, nowcasts
        from pysteps.utils import transformation
    except ImportError as exc:  # pragma: no cover - depends on optional compiled package
        raise RuntimeError(
            "PySTEPS is not installed. On Windows/Python 3.12 it may require Microsoft C++ "
            "Build Tools; conda-forge on Python 3.11 is the recommended route."
        ) from exc
    return motion, nowcasts, transformation


def pysteps_deterministic_extrapolation(
    history: np.ndarray,
    lead_steps: int,
    *,
    rain_threshold: float = 0.1,
    zerovalue: float = -15.0,
) -> np.ndarray:
    """Run PySTEPS Lucas-Kanade plus deterministic extrapolation on rain rates.

    The wrapper follows the public PySTEPS example: dB-transform recent rain-rate fields,
    estimate Lucas-Kanade motion from the transformed history, advect the latest field, and
    invert back to mm/h.
    """
    if lead_steps < 1:
        raise ValueError("lead_steps must be positive")
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("history must have shape (at least 3, y, x)")
    motion, nowcasts, transformation = _pysteps_modules()
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    transformed, metadata = transformation.dB_transform(
        finite,
        metadata={"unit": "mm/h", "transform": None},
        threshold=rain_threshold,
        zerovalue=zerovalue,
    )
    transformed[~np.isfinite(transformed)] = metadata.get("zerovalue", zerovalue)
    oflow = motion.get_method("LK")
    velocity = oflow(transformed[-3:])
    extrapolate = nowcasts.get_method("extrapolation")
    forecast = extrapolate(transformed[-1], velocity, lead_steps)
    result, _ = transformation.dB_transform(forecast, metadata=metadata, inverse=True)
    return np.clip(np.asarray(result, dtype=np.float32), 0.0, None)


def pysteps_steps_ensemble_mean(
    history: np.ndarray,
    lead_steps: int,
    *,
    km_per_pixel: float,
    timestep_minutes: float,
    ensemble_members: int = 8,
    rain_threshold: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Run a small PySTEPS STEPS ensemble and return its ensemble mean in mm/h."""
    if lead_steps < 1:
        raise ValueError("lead_steps must be positive")
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("history must have shape (at least 3, y, x)")
    motion, nowcasts, transformation = _pysteps_modules()
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    transformed, metadata = transformation.dB_transform(
        finite,
        metadata={"unit": "mm/h", "transform": None},
        threshold=rain_threshold,
        zerovalue=-15.0,
    )
    oflow = motion.get_method("LK")
    velocity = oflow(transformed[-3:])
    steps = nowcasts.get_method("steps")
    with contextlib.redirect_stdout(io.StringIO()):
        ensemble = steps(
            transformed,
            velocity,
            lead_steps,
            n_ens_members=ensemble_members,
            precip_thr=rain_threshold,
            kmperpixel=km_per_pixel,
            timestep=timestep_minutes,
            seed=seed,
            num_workers=1,
            fft_method="numpy",
            measure_time=False,
        )
    mean_transformed = np.nanmean(ensemble, axis=0)
    result, _ = transformation.dB_transform(mean_transformed, metadata=metadata, inverse=True)
    return np.clip(np.asarray(result, dtype=np.float32), 0.0, None)


def pysteps_steps_ensemble(
    history: np.ndarray,
    lead_steps: int,
    *,
    km_per_pixel: float,
    timestep_minutes: float,
    ensemble_members: int = 12,
    rain_threshold: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Return reproducible STEPS members in mm/h, shaped ``(member, lead, y, x)``."""
    if lead_steps < 1 or ensemble_members < 2:
        raise ValueError("lead_steps must be positive and ensemble_members at least two")
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("history must have shape (at least 3, y, x)")
    motion, nowcasts, transformation = _pysteps_modules()
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    transformed, metadata = transformation.dB_transform(
        finite,
        metadata={"unit": "mm/h", "transform": None},
        threshold=rain_threshold,
        zerovalue=-15.0,
    )
    transformed[~np.isfinite(transformed)] = metadata.get("zerovalue", -15.0)
    velocity = motion.get_method("LK")(transformed[-3:])
    steps = nowcasts.get_method("steps")
    with contextlib.redirect_stdout(io.StringIO()):
        ensemble = steps(
            transformed,
            velocity,
            lead_steps,
            n_ens_members=ensemble_members,
            precip_thr=rain_threshold,
            kmperpixel=km_per_pixel,
            timestep=timestep_minutes,
            seed=seed,
            num_workers=1,
            fft_method="numpy",
            measure_time=False,
        )
    result, _ = transformation.dB_transform(ensemble, metadata=metadata, inverse=True)
    return np.clip(np.asarray(result, dtype=np.float32), 0.0, None)
