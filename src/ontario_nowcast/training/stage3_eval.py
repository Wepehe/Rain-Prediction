"""Evaluate Stage 3 learned and conventional baselines on identical tile samples."""

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
from .stage3_data import Stage3RadarDataset, collate_stage3

LEAD_STEPS = (1, 2, 3, 5, 8, 10, 15, 20)
THRESHOLDS = (0.1, 1.0, 2.5, 5.0)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _metric_bucket() -> dict[str, Any]:
    return {
        "hits": 0,
        "misses": 0,
        "false_alarms": 0,
        "correct_negatives": 0,
        "fss_radius_18km_values": [],
        "brier_values": [],
    }


def _update_bucket(
    bucket: dict[str, Any],
    observed: np.ndarray,
    forecast: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    resolution_km: float,
) -> None:
    metrics = categorical_metrics(observed, forecast, threshold)
    for key in ("hits", "misses", "false_alarms", "correct_negatives"):
        bucket[key] += int(metrics[key])
    bucket["fss_radius_18km_values"].append(
        fractions_skill_score_km(observed, forecast, threshold, 18.0, resolution_km)
    )
    bucket["brier_values"].append(brier_score((observed >= threshold).astype(float), probability))


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, float | int]:
    h, m, f, c = (
        bucket["hits"],
        bucket["misses"],
        bucket["false_alarms"],
        bucket["correct_negatives"],
    )

    def ratio(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")

    return {
        "hits": h,
        "misses": m,
        "false_alarms": f,
        "correct_negatives": c,
        "csi": ratio(h, h + m + f),
        "pod": ratio(h, h + m),
        "far": ratio(f, h + f),
        "f1": ratio(2 * h, 2 * h + f + m),
        "fss_radius_18km": float(np.nanmean(bucket["fss_radius_18km_values"])),
        "brier_score": float(np.nanmean(bucket["brier_values"])),
    }


def _recover_history(batch: dict[str, Any], normalization: dict[str, float]) -> np.ndarray:
    inputs = batch["inputs"].detach().cpu().numpy()
    mean = float(normalization["mean"])
    std = float(normalization["std"])
    history_log = inputs[:, :, 0] * std + mean
    mask = inputs[:, :, 1] > 0
    history = np.expm1(history_log)
    return np.where(mask, history, np.nan).astype(np.float32)


def _observed_targets(batch: dict[str, Any]) -> np.ndarray:
    target_log = batch["target_intensity"].detach().cpu().numpy()
    mask = batch["target_mask"].detach().cpu().numpy() > 0
    return np.where(mask, np.expm1(target_log), np.nan).astype(np.float32)


def _make_model(config: dict[str, Any], checkpoint: Path, device: Any) -> Any:
    import torch

    model = build_radar_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def _probability_for_threshold(
    occurrence_probability: np.ndarray, model_cfg: dict[str, Any], threshold: float
) -> np.ndarray:
    if occurrence_probability.ndim == 3:
        return occurrence_probability
    thresholds = [float(value) for value in model_cfg.get("occurrence_thresholds_mm_hr", [0.1])]
    if threshold in thresholds:
        threshold_index = thresholds.index(threshold)
    else:
        threshold_index = int(np.argmin(np.abs(np.asarray(thresholds) - threshold)))
    return occurrence_probability[:, threshold_index]


def evaluate_stage3(
    config_path: Path,
    *,
    manifest_path: Path = Path("artifacts/stage_3/sample_manifest.csv"),
    checkpoint: Path = Path("artifacts/stage_3/model_best.pt"),
    normalization_path: Path = Path("artifacts/stage_3/normalization.json"),
    output_dir: Path = Path("artifacts/stage_3"),
    split: str = "dev",
    max_samples: int | None = None,
    include_pysteps: bool = False,
) -> dict[str, Any]:
    import torch

    config = _load_yaml(config_path)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    dataset = Stage3RadarDataset(manifest_path, split=split, normalization=normalization)
    if max_samples is not None:
        dataset.table = dataset.table.head(max_samples).reset_index(drop=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_stage3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(config, checkpoint, device)
    buckets: dict[tuple[str, int, float], dict[str, Any]] = defaultdict(_metric_bucket)
    event_buckets: dict[tuple[str, str, int, float], dict[str, Any]] = defaultdict(_metric_bucket)
    resolution_km = float(config["data"]["grid_resolution_km"])
    for index, batch in enumerate(loader, start=1):
        event_id = str(batch["metadata"][0]["event_id"])
        model_batch = {
            key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
        }
        with torch.no_grad():
            occurrence_logits, intensity_raw = model(model_batch["inputs"])
            learned_rate = expected_rate_mm_hr(occurrence_logits, intensity_raw).detach().cpu().numpy()[0]
            learned_probability = torch.sigmoid(occurrence_logits).detach().cpu().numpy()[0]
        observed = _observed_targets(batch)[0]
        history = _recover_history(batch, normalization)[0]
        forecasts = {
            "learned_convlstm": learned_rate,
            "persistence": persistence(history[-1], int(config["data"]["target_frames"])),
            "optical_flow": optical_flow_extrapolation(
                history[-2], history[-1], int(config["data"]["target_frames"])
            ),
        }
        if include_pysteps:
            forecasts["pysteps_extrapolation"] = pysteps_deterministic_extrapolation(
                history[-3:], int(config["data"]["target_frames"])
            )
        for lead_step in LEAD_STEPS:
            lead_minutes = lead_step * int(config["data"]["cadence_minutes"])
            obs = observed[lead_step - 1]
            for model_name, forecast in forecasts.items():
                pred = forecast[lead_step - 1]
                for threshold in THRESHOLDS:
                    probability = (
                        _probability_for_threshold(
                            learned_probability, config["model"], float(threshold)
                        )[lead_step - 1]
                        if model_name == "learned_convlstm"
                        else (pred >= threshold).astype(float)
                    )
                    _update_bucket(
                        buckets[(model_name, lead_minutes, threshold)],
                        obs,
                        pred,
                        probability,
                        threshold=threshold,
                        resolution_km=resolution_km,
                    )
                    _update_bucket(
                        event_buckets[(event_id, model_name, lead_minutes, threshold)],
                        obs,
                        pred,
                        probability,
                        threshold=threshold,
                        resolution_km=resolution_km,
                    )
        if index == 1 or index % 10 == 0:
            print(f"[stage3-eval] {split}: {index}/{len(dataset)} samples", flush=True)
    rows = [
        {"model": model_name, "lead_minutes": lead, "threshold_mm_hr": threshold, **_finalize_bucket(bucket)}
        for (model_name, lead, threshold), bucket in buckets.items()
    ]
    event_rows = [
        {
            "event_id": event_id,
            "model": model_name,
            "lead_minutes": lead,
            "threshold_mm_hr": threshold,
            **_finalize_bucket(bucket),
        }
        for (event_id, model_name, lead, threshold), bucket in event_buckets.items()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{split}_metrics.csv"
    event_metrics_path = output_dir / f"{split}_event_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(event_rows).to_csv(event_metrics_path, index=False)
    summary = {
        "split": split,
        "samples": len(dataset),
        "include_pysteps": include_pysteps,
        "metrics": metrics_path.as_posix(),
        "event_metrics": event_metrics_path.as_posix(),
    }
    (output_dir / f"{split}_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/stage_3_radar_only.yaml")
    )
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/stage_3/sample_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/stage_3/model_best.pt"))
    parser.add_argument(
        "--normalization", type=Path, default=Path("artifacts/stage_3/normalization.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_3"))
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "existing_test", "fresh_holdout"],
        default="dev",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--include-pysteps", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_stage3(
                args.config,
                manifest_path=args.manifest,
                checkpoint=args.checkpoint,
                normalization_path=args.normalization,
                output_dir=args.output_dir,
                split=args.split,
                max_samples=args.max_samples,
                include_pysteps=args.include_pysteps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
