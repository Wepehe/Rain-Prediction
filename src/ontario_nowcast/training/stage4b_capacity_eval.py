"""Evaluate the Stage 4B radar-only A+ capacity control on DEV tensors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..evaluation.metrics import brier_score, categorical_metrics, fractions_skill_score_km
from ..models.baselines import (
    optical_flow_extrapolation,
    persistence,
    pysteps_deterministic_extrapolation,
)
from ..models.radar_convlstm import build_radar_model, expected_rate_mm_hr
from .stage4b_eval import LEAD_STEPS, THRESHOLDS, _finalize_bucket, _metric_bucket
from .stage4b_train import HOLDOUT_SPLITS, _load_npz


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _radar_input(payload: dict[str, np.ndarray], normalization: dict[str, Any]) -> np.ndarray:
    radar = payload["radar"]
    mask = payload["radar_mask"]
    radar_log = np.where(mask > 0, np.log1p(np.nan_to_num(radar, nan=0.0)), 0.0)
    mean = float(normalization["radar"]["mean"])
    std = max(float(normalization["radar"]["std"]), 1e-6)
    return np.stack([(radar_log - mean) / std, mask], axis=1).astype(np.float32)


def _observed(payload: dict[str, np.ndarray]) -> np.ndarray:
    target = payload["target"]
    mask = payload["target_mask"] > 0
    return np.where(mask, target, np.nan).astype(np.float32)


def _update_bucket(
    bucket: dict[str, Any],
    observed: np.ndarray,
    forecast: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    resolution_km: float = 2.0,
) -> None:
    metrics = categorical_metrics(observed, forecast, threshold)
    for key in ("hits", "misses", "false_alarms", "correct_negatives"):
        bucket[key] += int(metrics[key])
    bucket["fss_radius_18km_values"].append(
        fractions_skill_score_km(observed, forecast, threshold, 18.0, resolution_km)
    )
    bucket["brier_values"].append(brier_score((observed >= threshold).astype(float), probability))


def evaluate_capacity_control(
    *,
    config_path: Path = Path("configs/experiments/stage_4b_a_plus_radar_capacity.yaml"),
    checkpoint: Path = Path("artifacts/stage_4b/a_plus_radar/model_best.pt"),
    metadata_path: Path = Path("artifacts/stage_4b/a_plus_radar/training_metadata.json"),
    tensor_manifest: Path = Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
    output_dir: Path = Path("artifacts/stage_4b/a_plus_radar/evaluation_pysteps"),
    include_pysteps: bool = True,
) -> dict[str, Any]:
    import torch

    config = _load_yaml(config_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(tensor_manifest)
    dev = manifest[manifest["split"].astype(str).isin({"dev", "stage4_dev_repair"})].copy()
    if dev["split"].astype(str).isin(HOLDOUT_SPLITS).any():
        raise ValueError("A+ DEV evaluation refuses final initiation holdout")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_radar_model({"model": config["model"], "data": config["data"]}).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    buckets: dict[tuple[str, int, float], dict[str, Any]] = defaultdict(_metric_bucket)
    for index, row in enumerate(dev.itertuples(index=False), start=1):
        payload = _load_npz(Path(str(row.tensor_path)))
        observed = _observed(payload)
        radar = torch.from_numpy(_radar_input(payload, metadata["normalization"])[None]).to(device)
        with torch.no_grad():
            logits, intensity = model(radar)
            learned_rate = expected_rate_mm_hr(logits, intensity).detach().cpu().numpy()[0]
            learned_probability = torch.sigmoid(logits).detach().cpu().numpy()[0]
        history = payload["radar"]
        forecasts = {
            "a_plus_radar": (learned_rate, learned_probability),
            "persistence": (persistence(history[-1], 20), None),
            "optical_flow": (optical_flow_extrapolation(history[-2], history[-1], 20), None),
        }
        if include_pysteps:
            forecasts["pysteps_extrapolation"] = (
                pysteps_deterministic_extrapolation(history[-3:], 20),
                None,
            )
        for lead_step in LEAD_STEPS:
            lead_minutes = lead_step * 6
            obs = observed[lead_step - 1]
            for model_name, (forecast, probability_raw) in forecasts.items():
                pred = forecast[lead_step - 1]
                for threshold_index, threshold in enumerate(THRESHOLDS):
                    probability = (
                        probability_raw[lead_step - 1, threshold_index]
                        if probability_raw is not None
                        else (pred >= threshold).astype(float)
                    )
                    _update_bucket(
                        buckets[(model_name, lead_minutes, threshold)],
                        obs,
                        pred,
                        probability,
                        threshold=threshold,
                    )
        if index == 1 or index % 5 == 0:
            print(f"[stage4b-capacity-eval] {index}/{len(dev)} DEV tensors", flush=True)
    rows = [
        {"model": model_name, "lead_minutes": lead, "threshold_mm_hr": threshold, **_finalize_bucket(bucket)}
        for (model_name, lead, threshold), bucket in sorted(buckets.items())
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "dev_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    summary = {
        "config": config_path.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "samples": len(dev),
        "include_pysteps": include_pysteps,
        "final_holdout_scoring": "not_evaluated",
        "metrics": metrics_path.as_posix(),
    }
    (output_dir / "dev_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pysteps", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate_capacity_control(include_pysteps=not args.no_pysteps), indent=2))


if __name__ == "__main__":
    main()
