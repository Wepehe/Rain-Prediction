"""Frozen operational inference for validated Cycle-2 Residual V1.

This module deliberately has no dependency on research split manifests. It verifies
the portable bundle before loading and reproduces the validated preprocessing,
PySTEPS baseline, neural correction, and 0.35 occurrence decision.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..models.pysteps_residual_unet import PySTEPSResidualUNetV1, count_parameters
from ..training.cycle2_residual import deterministic_pysteps_with_motion

DEFAULT_BUNDLE = Path("artifacts/operational/residual_v1")
MODEL_VERSION = "cycle2-residual-v1"
OCCURRENCE_THRESHOLD = 0.35
RAIN_THRESHOLD_MM_H = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_timestamps(values: Sequence[Any]) -> list[datetime]:
    if len(values) != 10:
        raise ValueError(f"timestamps must contain exactly 10 values; got {len(values)}")
    parsed: list[datetime] = []
    for value in values:
        if isinstance(value, datetime):
            item = value
        else:
            text = str(value).replace("Z", "+00:00")
            try:
                item = datetime.fromisoformat(text)
            except ValueError as exc:
                raise ValueError(f"invalid timestamp: {value!r}") from exc
        parsed.append(item)
    if any(b <= a for a, b in pairwise(parsed)):
        raise ValueError("timestamps must be strictly increasing (oldest to newest)")
    if any((b - a) != timedelta(minutes=6) for a, b in pairwise(parsed)):
        raise ValueError("radar history must have an exact six-minute cadence")
    return parsed


@dataclass(frozen=True)
class OperationalForecast:
    timestamps: tuple[datetime, ...]
    pysteps_rate_mm_h: np.ndarray
    residual_probability: np.ndarray
    residual_wet_mask: np.ndarray
    residual_rate_mm_h: np.ndarray
    validity_mask: np.ndarray
    motion_u: np.ndarray | None
    motion_v: np.ndarray | None
    metadata: dict[str, Any]

    def save_npz(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "timestamps": np.asarray([x.isoformat() for x in self.timestamps]),
            "pysteps_rate_mm_h": self.pysteps_rate_mm_h,
            "residual_probability": self.residual_probability,
            "residual_wet_mask": self.residual_wet_mask,
            "residual_rate_mm_h": self.residual_rate_mm_h,
            "validity_mask": self.validity_mask,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        if self.motion_u is not None:
            payload["motion_u"] = self.motion_u
            payload["motion_v"] = self.motion_v
        grid = self.metadata.get("grid", {})
        if "x" in grid:
            payload["x"] = np.asarray(grid["x"])
        if "y" in grid:
            payload["y"] = np.asarray(grid["y"])
        np.savez_compressed(destination, **payload)
        return destination

    def summary(self) -> dict[str, Any]:
        axes = (1, 2)
        return {
            "initialization_timestamp": self.metadata["initialization_timestamp"],
            "model_version": self.metadata["model_version"],
            "forecast_timestamps": [x.isoformat() for x in self.timestamps],
            "lead_minutes": list(range(6, 121, 6)),
            "mean_wet_probability": self.residual_probability.mean(axis=axes).tolist(),
            "thresholded_wet_area_fraction": self.residual_wet_mask.mean(axis=axes).tolist(),
            "mean_corrected_rate_mm_h": self.residual_rate_mm_h.mean(axis=axes).tolist(),
            "maximum_corrected_rate_mm_h": self.residual_rate_mm_h.max(axis=axes).tolist(),
            "checkpoint_sha256": self.metadata["checkpoint_sha256"],
            "normalization_sha256": self.metadata["normalization_sha256"],
            "config_sha256": self.metadata["config_sha256"],
            "operating_threshold": self.metadata["operating_threshold"],
        }


@dataclass
class OperationalModel:
    model: PySTEPSResidualUNetV1
    device: torch.device
    manifest: dict[str, Any]
    bundle_dir: Path


def load_operational_model(
    bundle_dir: str | Path = DEFAULT_BUNDLE, *, device: str = "cpu"
) -> OperationalModel:
    bundle = Path(bundle_dir)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"operational manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, filename in {
        "checkpoint_sha256": "checkpoint.pt",
        "normalization_sha256": "normalization.json",
        "config_sha256": "config.yaml",
        "model_source_sha256": "model_source.py",
    }.items():
        path = bundle / filename
        if not path.is_file():
            raise FileNotFoundError(f"frozen operational artifact is missing: {path}")
        actual = _sha256(path)
        if actual != manifest[key]:
            raise RuntimeError(f"{filename} hash mismatch: expected {manifest[key]}, got {actual}")
    if float(manifest["operating_threshold"]) != OCCURRENCE_THRESHOLD:
        raise RuntimeError("bundle occurrence threshold does not match frozen 0.35 threshold")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    normalization = json.loads((bundle / "normalization.json").read_text(encoding="utf-8"))
    model = PySTEPSResidualUNetV1()
    if count_parameters(model) != int(manifest["trainable_parameters"]):
        raise RuntimeError("model architecture/parameter count does not match bundle")
    model.set_normalization(
        rate_mean=normalization["rate_log1p_mean"],
        rate_std=normalization["rate_log1p_std"],
        motion_mean=normalization["motion_mean"],
        motion_std=normalization["motion_std"],
    )
    try:
        checkpoint = torch.load(bundle / "checkpoint.pt", map_location=target, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    except Exception as exc:
        raise RuntimeError(f"could not load frozen checkpoint: {exc}") from exc
    model.to(target).eval()
    return OperationalModel(model=model, device=target, manifest=manifest, bundle_dir=bundle)


def forecast(
    operational_model: OperationalModel,
    history_rate: np.ndarray,
    timestamps: Sequence[Any],
    validity_mask: np.ndarray,
    metadata: dict[str, Any] | None = None,
    *,
    diagnostics: bool = False,
) -> OperationalForecast:
    history = np.asarray(history_rate)
    validity = np.asarray(validity_mask)
    if history.shape != (10, 128, 128):
        raise ValueError(f"history_rate must have shape (10, 128, 128); got {history.shape}")
    if validity.shape != history.shape:
        raise ValueError(f"validity_mask must have shape {history.shape}; got {validity.shape}")
    if not np.issubdtype(history.dtype, np.number):
        raise TypeError("history_rate must be numeric")
    validity = validity.astype(bool, copy=False) & np.isfinite(history)
    if not validity.any():
        raise ValueError("radar history is entirely invalid")
    parsed = _parse_timestamps(timestamps)
    clean = np.where(validity, np.maximum(history.astype(np.float32), 0.0), 0.0)
    start = time.perf_counter()
    pysteps_rate, motion, fallback = deterministic_pysteps_with_motion(clean)
    pysteps_seconds = time.perf_counter() - start
    inputs = (
        torch.from_numpy(clean)[None].to(operational_model.device),
        torch.from_numpy(pysteps_rate)[None].to(operational_model.device),
        torch.from_numpy(motion)[None].to(operational_model.device),
        torch.from_numpy(validity.astype(np.float32))[None].to(operational_model.device),
    )
    neural_start = time.perf_counter()
    with torch.inference_mode():
        output = operational_model.model(*inputs)
        probability = torch.sigmoid(output["occurrence_logits"])[0].cpu().numpy()
        corrected_rate = output["corrected_rate"][0].cpu().numpy()
    neural_seconds = time.perf_counter() - neural_start
    future = tuple(parsed[-1] + timedelta(minutes=6 * i) for i in range(1, 21))
    supplied = dict(metadata or {})
    product_metadata = {
        **supplied,
        "initialization_timestamp": parsed[-1].isoformat(),
        "model_version": operational_model.manifest["model_version"],
        "checkpoint_sha256": operational_model.manifest["checkpoint_sha256"],
        "normalization_sha256": operational_model.manifest["normalization_sha256"],
        "config_sha256": operational_model.manifest["config_sha256"],
        "model_source_sha256": operational_model.manifest["model_source_sha256"],
        "operating_threshold": OCCURRENCE_THRESHOLD,
        "rain_threshold_mm_h": RAIN_THRESHOLD_MM_H,
        "pysteps_fallback": fallback,
        "timing_seconds": {
            "pysteps": pysteps_seconds,
            "neural": neural_seconds,
            "total": pysteps_seconds + neural_seconds,
        },
    }
    return OperationalForecast(
        timestamps=future,
        pysteps_rate_mm_h=pysteps_rate.astype(np.float32),
        residual_probability=probability.astype(np.float32),
        residual_wet_mask=(probability >= OCCURRENCE_THRESHOLD),
        residual_rate_mm_h=corrected_rate.astype(np.float32),
        validity_mask=validity,
        motion_u=motion[0].astype(np.float32) if diagnostics else None,
        motion_v=motion[1].astype(np.float32) if diagnostics else None,
        metadata=product_metadata,
    )
