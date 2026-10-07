"""Evaluate Stage 4B radar+GOES ablations without touching the final holdout."""

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
from ..models.radar_convlstm import expected_rate_mm_hr
from ..models.radar_goes import build_stage4b_model
from .stage4b_train import HOLDOUT_SPLITS, Stage4BRadarGoesDataset, collate_stage4b

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


def _observed_targets(batch: dict[str, Any]) -> np.ndarray:
    target_log = batch["target_intensity"].detach().cpu().numpy()
    mask = batch["target_mask"].detach().cpu().numpy() > 0
    return np.where(mask, np.expm1(target_log), np.nan).astype(np.float32)


def _recover_history(batch: dict[str, Any], normalization: dict[str, Any]) -> np.ndarray:
    radar_inputs = batch["radar_inputs"].detach().cpu().numpy()
    mean = float(normalization["radar"]["mean"])
    std = float(normalization["radar"]["std"])
    history_log = radar_inputs[:, :, 0] * std + mean
    mask = radar_inputs[:, :, 1] > 0
    history = np.expm1(history_log)
    return np.where(mask, history, np.nan).astype(np.float32)


def _probability_for_threshold(
    occurrence_probability: np.ndarray,
    thresholds: list[float],
    threshold: float,
) -> np.ndarray:
    if occurrence_probability.ndim == 2:
        return occurrence_probability
    threshold_index = thresholds.index(threshold) if threshold in thresholds else 0
    return occurrence_probability[threshold_index]


def evaluate_stage4b(
    config_path: Path,
    *,
    checkpoint: Path,
    training_metadata: Path,
    output_dir: Path,
    tensor_manifest: Path = Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
    splits: list[str] | None = None,
    include_pysteps: bool = False,
) -> dict[str, Any]:
    import torch

    split_set = set(splits or ["dev", "stage4_dev_repair"])
    if split_set & HOLDOUT_SPLITS:
        raise ValueError("Stage 4B evaluation refuses to score final initiation holdout before freeze")
    config = _load_yaml(config_path)
    metadata = json.loads(training_metadata.read_text(encoding="utf-8"))
    manifest = pd.read_csv(tensor_manifest)
    thresholds = [float(value) for value in config["model"]["occurrence_thresholds_mm_hr"]]
    dataset = Stage4BRadarGoesDataset(
        manifest,
        splits=split_set,
        radar_normalization=metadata["normalization"]["radar"],
        goes_normalization=metadata["normalization"]["goes"],
        goes_channel_indices=[int(value) for value in config["data"]["goes_channel_indices"]],
        rain_threshold_mm_hr=float(config["data"].get("rain_threshold_mm_hr", 0.1)),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_stage4b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_stage4b_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    buckets: dict[tuple[str, int, float], dict[str, Any]] = defaultdict(_metric_bucket)
    hard_negative_rows = []
    resolution_km = 2.0
    for index, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
        observed_batch = _observed_targets(batch)
        history_batch = _recover_history(batch, metadata["normalization"])
        variants = {
            "normal_goes": batch["goes_inputs"],
            "zeroed_goes": torch.zeros_like(batch["goes_inputs"]),
            "shuffled_goes": batch["goes_inputs"].flip(0)
            if batch["goes_inputs"].shape[0] > 1
            else torch.roll(batch["goes_inputs"], 1, 2),
        }
        forecasts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        with torch.no_grad():
            for name, goes_inputs in variants.items():
                occurrence_logits, intensity_raw = model(batch["radar_inputs"], goes_inputs)
                rate = expected_rate_mm_hr(occurrence_logits, intensity_raw).detach().cpu().numpy()
                probability = torch.sigmoid(occurrence_logits).detach().cpu().numpy()
                forecasts[name] = (rate, probability)
        for sample_index, metadata_row in enumerate(batch["metadata"]):
            observed = observed_batch[sample_index]
            history = history_batch[sample_index]
            deterministic_forecasts = {
                "persistence": persistence(history[-1], 20),
                "optical_flow": optical_flow_extrapolation(history[-2], history[-1], 20),
            }
            if include_pysteps:
                deterministic_forecasts["pysteps_extrapolation"] = pysteps_deterministic_extrapolation(
                    history[-3:], 20
                )
            for lead_step in LEAD_STEPS:
                lead_minutes = lead_step * 6
                obs = observed[lead_step - 1]
                for model_name, forecast in deterministic_forecasts.items():
                    pred = forecast[lead_step - 1]
                    for threshold in THRESHOLDS:
                        _update_bucket(
                            buckets[(model_name, lead_minutes, threshold)],
                            obs,
                            pred,
                            (pred >= threshold).astype(float),
                            threshold=threshold,
                            resolution_km=resolution_km,
                        )
                for model_name, (rate, probability) in forecasts.items():
                    pred = rate[sample_index, lead_step - 1]
                    for threshold in THRESHOLDS:
                        prob = _probability_for_threshold(
                            probability[sample_index, lead_step - 1], thresholds, float(threshold)
                        )
                        _update_bucket(
                            buckets[(model_name, lead_minutes, threshold)],
                            obs,
                            pred,
                            prob,
                            threshold=threshold,
                            resolution_km=resolution_km,
                        )
            if str(metadata_row.get("event_class")) == "hard_negative_candidate":
                prob_01 = forecasts["normal_goes"][1][sample_index, :, 0]
                hard_negative_rows.append(
                    {
                        "sample_id": metadata_row["anchor_id"],
                        "event_id": metadata_row["event_id"],
                        "mean_p_rain_0p1": float(np.nanmean(prob_01)),
                        "max_p_rain_0p1": float(np.nanmax(prob_01)),
                        "wet_area_prediction_fraction_p_ge_0p5": float(np.nanmean(prob_01 >= 0.5)),
                        "brier_score_0p1": brier_score(
                            (observed >= 0.1).astype(float),
                            prob_01,
                        ),
                    }
                )
        if index == 1 or index % 5 == 0:
            print(f"[stage4b-eval] {index}/{len(loader)} batches", flush=True)
    rows = [
        {"model": model_name, "lead_minutes": lead, "threshold_mm_hr": threshold, **_finalize_bucket(bucket)}
        for (model_name, lead, threshold), bucket in sorted(buckets.items())
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "dev_metrics.csv"
    hard_negative_path = output_dir / "dev_hard_negative_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(hard_negative_rows).to_csv(hard_negative_path, index=False)
    summary = {
        "config": config_path.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "splits": sorted(split_set),
        "samples": len(dataset),
        "include_pysteps": include_pysteps,
        "final_holdout_scoring": "not_evaluated",
        "metrics": metrics_path.as_posix(),
        "hard_negative_metrics": hard_negative_path.as_posix(),
    }
    (output_dir / "dev_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tensor-manifest",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
    )
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--include-pysteps", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_stage4b(
                args.config,
                checkpoint=args.checkpoint,
                training_metadata=args.training_metadata,
                output_dir=args.output_dir,
                tensor_manifest=args.tensor_manifest,
                splits=args.splits,
                include_pysteps=args.include_pysteps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
