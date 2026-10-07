"""Validate H1 caches and derive TRAIN-only HRRR normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..models.frozen_residual_fusion import FrozenBaselineResidualFusion


def run(output_dir: Path = Path("artifacts/stage_5/gate")) -> dict[str, object]:
    manifest_path = Path("artifacts/stage_5/cache/stage5_cache_manifest.csv")
    table = pd.read_csv(manifest_path)
    if len(table) != 72 or not set(table.split.unique()) <= {"train", "dev", "stage4_dev_repair"}:
        raise RuntimeError("invalid Stage 5 cache membership")
    values = []
    integrity = []
    for row in table.itertuples(index=False):
        payload = np.load(row.stage5_cache_path)
        hrrr = payload["hrrr_apcp_hourly"]
        if row.split == "train":
            values.append(np.log1p(hrrr[np.isfinite(hrrr)]))
        integrity.append(
            payload["aplus_logits"].shape == (20, 4, 128, 128)
            and payload["aplus_intensity_raw"].shape == (20, 128, 128)
            and payload["aplus_expected_rate_mm_hr"].shape == (20, 128, 128)
            and hrrr.shape == (2, 128, 128)
            and str(payload["aplus_checkpoint_sha256"])
            == "63e049c5ea75c3aa6af7b461fe6e9bca2db3c51198a667e71f02a7c6217efe70"
        )
    train_values = np.concatenate(values)
    normalization = {
        "source_split": "train",
        "transform": "log1p",
        "mean": float(train_values.mean()),
        "std": float(train_values.std()),
        "count": int(train_values.size),
        "units": "kg m-2 per one-hour interval",
        "cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hrrr_apcp_normalization_train_only.json").write_text(
        json.dumps(normalization, indent=2)
    )
    model = FrozenBaselineResidualFusion()
    base_logits = torch.randn(1, 20, 4, 32, 32)
    base_intensity = torch.randn(1, 20, 32, 32)
    base_expected = torch.rand(1, 20, 32, 32)
    hrrr = torch.randn(1, 2, 2, 32, 32)
    goes = torch.randn(1, 10, 10, 32, 32)
    with torch.no_grad():
        logits, intensity, raw = model(base_logits, base_intensity, base_expected, hrrr, goes)
    zero_identity = bool(
        torch.equal(logits, base_logits)
        and torch.equal(intensity, base_intensity)
        and torch.count_nonzero(raw) == 0
    )
    result = {
        "cache_rows": len(table),
        "all_cache_shapes_and_hashes_valid": all(integrity),
        "train_only_hrrr_normalization": normalization,
        "zero_initialized_exact_a_plus_identity": zero_identity,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "frozen_aplus_parameters_in_model": 0,
        "protected_holdout_used": False,
        "training_started": False,
    }
    (output_dir / "stage5_pretraining_gate.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
