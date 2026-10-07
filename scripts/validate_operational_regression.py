"""Freeze and validate the operational path against one non-FINAL DEV cache row."""
from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from ontario_nowcast.operational.inference import forecast, load_operational_model
from ontario_nowcast.training.cycle2_residual import deterministic_pysteps_with_motion

DEV_CACHE = Path(
    "artifacts/cycle2/residual_v1/cache/new_dev/"
    "cycle2_20201019__active__20201019T1254__11.npz"
)
FIXTURE = Path("tests/fixtures/operational_residual_v1")


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    with np.load(DEV_CACHE, allow_pickle=False) as cached:
        history = cached["history"].astype(np.float32)
        validity = cached["validity"].astype(bool)
        cached_pysteps = cached["pysteps"].astype(np.float32)
        cached_motion = cached["motion"].astype(np.float32)
    issue = datetime.fromisoformat("2020-10-19T12:54:00+00:00")
    timestamps = [issue - timedelta(minutes=6 * i) for i in range(9, -1, -1)]
    model = load_operational_model(device="cpu")
    tracemalloc.start()
    started = time.perf_counter()
    result = forecast(model, history, timestamps, validity, diagnostics=True)
    wall = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Research-path reference: the same fresh float32 generation used by FINAL evaluation.
    clean = np.where(validity, history, 0.0).astype(np.float32)
    reference_pysteps, reference_motion, _ = deterministic_pysteps_with_motion(clean)
    with torch.inference_mode():
        reference = model.model(
            torch.from_numpy(clean)[None],
            torch.from_numpy(reference_pysteps)[None],
            torch.from_numpy(reference_motion)[None],
            torch.from_numpy(validity.astype(np.float32))[None],
        )
        reference_probability = torch.sigmoid(reference["occurrence_logits"])[0].numpy()
        reference_rate = reference["corrected_rate"][0].numpy()
    differences = {
        "pysteps_max_abs": float(np.max(np.abs(result.pysteps_rate_mm_h - reference_pysteps))),
        "probability_max_abs": float(np.max(np.abs(result.residual_probability - reference_probability))),
        "corrected_rate_max_abs": float(np.max(np.abs(result.residual_rate_mm_h - reference_rate))),
        "wet_mask_different_pixels": int(np.count_nonzero(
            result.residual_wet_mask != (reference_probability >= 0.35)
        )),
    }
    cache_quantization_differences = {
        "pysteps_max_abs": float(np.max(np.abs(reference_pysteps - cached_pysteps))),
        "motion_max_abs": float(np.max(np.abs(reference_motion - cached_motion))),
    }
    if differences["pysteps_max_abs"] > 2e-3:
        raise AssertionError(differences)
    if differences["probability_max_abs"] > 2e-5 or differences["corrected_rate_max_abs"] > 2e-4:
        raise AssertionError(differences)
    if differences["wet_mask_different_pixels"]:
        raise AssertionError(differences)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    input_path = FIXTURE / "dev_input.npz"
    np.savez_compressed(
        input_path,
        history_rate=history,
        timestamps=np.asarray([x.isoformat() for x in timestamps]),
        validity_mask=validity,
        metadata_json=np.asarray(json.dumps({"projection": "EPSG:3978", "resolution_km": 2})),
    )
    output_hash = array_hash(
        result.pysteps_rate_mm_h,
        result.residual_probability,
        result.residual_wet_mask,
        result.residual_rate_mm_h,
    )
    report = {
        "purpose": "software regression only; not model tuning",
        "split": "new_dev",
        "row_id": "cycle2_20201019__active__20201019T1254__11",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_array_sha256": output_hash,
        "differences_from_research_path": differences,
        "detached_float16_cache_quantization_differences": cache_quantization_differences,
        "probability_summary": {
            "mean": float(result.residual_probability.mean()),
            "maximum": float(result.residual_probability.max()),
        },
        "rate_summary_mm_h": {
            "mean": float(result.residual_rate_mm_h.mean()),
            "maximum": float(result.residual_rate_mm_h.max()),
        },
        "cpu_benchmark": {
            **result.metadata["timing_seconds"],
            "wall": wall,
            "python_tracemalloc_peak_mb": peak / 1024**2,
            "tile_shape": [128, 128],
        },
    }
    (FIXTURE / "golden.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
