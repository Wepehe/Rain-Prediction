"""Stage 3.1 initiation and dissipation timing evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..evaluation.onset import first_crossing_minutes, onset_metrics
from ..models.baselines import (
    optical_flow_extrapolation,
    persistence,
    pysteps_deterministic_extrapolation,
)
from ..models.radar_convlstm import build_radar_model, expected_rate_mm_hr
from .stage3_data import Stage3RadarDataset, collate_stage3
from .stage3_eval import _observed_targets, _probability_for_threshold, _recover_history


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_model(config: dict[str, Any], checkpoint: Path, device: Any) -> Any:
    import torch

    model = build_radar_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def _append_metrics(
    rows: list[dict[str, Any]],
    *,
    split: str,
    sample_category: str,
    lifecycle: str,
    model_name: str,
    observed_minutes: np.ndarray,
    predicted_minutes: np.ndarray,
) -> None:
    metrics = onset_metrics(observed_minutes, predicted_minutes)
    observed_events = int(metrics["observed_events"])
    predicted_events = int(metrics["predicted_events"])
    missed_events = int(metrics["missed_events"])
    false_events = int(metrics["false_events"])
    paired_events = int(metrics["paired_events"])
    precision = paired_events / predicted_events if predicted_events else np.nan
    recall = paired_events / observed_events if observed_events else np.nan
    false_fraction = false_events / max(false_events + paired_events, 1)
    rows.append(
        {
            "split": split,
            "sample_category": sample_category,
            "lifecycle": lifecycle,
            "model": model_name,
            **metrics,
            "precision": precision,
            "recall": recall,
            "false_event_fraction_of_predictions": false_fraction,
            "miss_fraction_of_observed": missed_events / observed_events
            if observed_events
            else np.nan,
        }
    )
    for window_minutes in (30, 60, 90, 120):
        observed_in_window = np.isfinite(observed_minutes) & (observed_minutes <= window_minutes)
        predicted_in_window = np.isfinite(predicted_minutes) & (predicted_minutes <= window_minutes)
        hits = int(np.sum(observed_in_window & predicted_in_window))
        misses = int(np.sum(observed_in_window & ~predicted_in_window))
        false_alarms = int(np.sum(~observed_in_window & predicted_in_window))
        rows.append(
            {
                "split": split,
                "sample_category": sample_category,
                "lifecycle": f"{lifecycle}_by_{window_minutes}min",
                "model": model_name,
                "observed_events": int(np.sum(observed_in_window)),
                "predicted_events": int(np.sum(predicted_in_window)),
                "missed_events": misses,
                "false_events": false_alarms,
                "paired_events": hits,
                "median_error_minutes": np.nan,
                "mae_minutes": np.nan,
                "within_5_minutes": np.nan,
                "within_10_minutes": np.nan,
                "within_15_minutes": np.nan,
                "within_30_minutes": np.nan,
                "precision": hits / (hits + false_alarms) if hits + false_alarms else np.nan,
                "recall": hits / (hits + misses) if hits + misses else np.nan,
                "false_event_fraction_of_predictions": false_alarms / max(hits + false_alarms, 1),
                "miss_fraction_of_observed": misses / (hits + misses) if hits + misses else np.nan,
            }
        )


def evaluate_lifecycle(
    config_path: Path,
    *,
    manifest_path: Path,
    checkpoint: Path,
    normalization_path: Path,
    output_dir: Path,
    split: str,
    probability_threshold: float,
    include_pysteps: bool = True,
) -> dict[str, Any]:
    import torch

    config = _load_yaml(config_path)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    dataset = Stage3RadarDataset(manifest_path, split=split, normalization=normalization)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_stage3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(config, checkpoint, device)
    threshold = float(config["data"]["rain_threshold_mm_hr"])
    interval = int(config["data"]["cadence_minutes"])
    grouped: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for index, batch in enumerate(loader, start=1):
        metadata = batch["metadata"][0]
        category = str(metadata["category"])
        model_batch = {
            key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
        }
        with torch.no_grad():
            occurrence_logits, intensity_raw = model(model_batch["inputs"])
            learned_expected = expected_rate_mm_hr(occurrence_logits, intensity_raw).detach().cpu().numpy()[0]
            probability_raw = torch.sigmoid(occurrence_logits).detach().cpu().numpy()[0]
            learned_probability = _probability_for_threshold(
                probability_raw, config["model"], threshold
            )
        observed = _observed_targets(batch)[0]
        history = _recover_history(batch, normalization)[0]
        forecasts = {
            "learned_expected_rate": learned_expected,
            "learned_probability_head": learned_probability,
            "persistence": persistence(history[-1], int(config["data"]["target_frames"])),
            "optical_flow": optical_flow_extrapolation(
                history[-2], history[-1], int(config["data"]["target_frames"])
            ),
        }
        if include_pysteps:
            forecasts["pysteps_extrapolation"] = pysteps_deterministic_extrapolation(
                history[-3:], int(config["data"]["target_frames"])
            )
        issue = history[-1]
        onset_mask = np.isfinite(issue) & (issue < threshold)
        cessation_mask = np.isfinite(issue) & (issue >= threshold)
        observed_onset = first_crossing_minutes(observed, threshold, interval)
        observed_cessation = first_crossing_minutes(observed, threshold, interval, above=False)
        for model_name, forecast in forecasts.items():
            if model_name == "learned_probability_head":
                predicted_onset = first_crossing_minutes(forecast, probability_threshold, interval)
                predicted_cessation = first_crossing_minutes(
                    forecast, probability_threshold, interval, above=False
                )
            else:
                predicted_onset = first_crossing_minutes(forecast, threshold, interval)
                predicted_cessation = first_crossing_minutes(
                    forecast, threshold, interval, above=False
                )
            if category == "initiation_centered":
                grouped[(category, "initiation", model_name)].append(
                    (observed_onset[onset_mask], predicted_onset[onset_mask])
                )
            if category == "dissipation":
                grouped[(category, "dissipation", model_name)].append(
                    (observed_cessation[cessation_mask], predicted_cessation[cessation_mask])
                )
        if index == 1 or index % 25 == 0:
            print(f"[stage31-lifecycle] {split}: {index}/{len(dataset)} samples", flush=True)
    rows = []
    for (category, lifecycle, model_name), pairs in sorted(grouped.items()):
        observed_values = np.concatenate([pair[0].reshape(-1) for pair in pairs])
        predicted_values = np.concatenate([pair[1].reshape(-1) for pair in pairs])
        _append_metrics(
            rows,
            split=split,
            sample_category=category,
            lifecycle=lifecycle,
            model_name=model_name,
            observed_minutes=observed_values,
            predicted_minutes=predicted_values,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{split}_lifecycle_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    summary = {
        "split": split,
        "samples": len(dataset),
        "probability_threshold": probability_threshold,
        "include_pysteps": include_pysteps,
        "metrics": metrics_path.as_posix(),
    }
    (output_dir / f"{split}_lifecycle_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "existing_test", "fresh_holdout"], required=True)
    parser.add_argument("--probability-threshold", type=float, required=True)
    parser.add_argument("--no-pysteps", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_lifecycle(
                args.config,
                manifest_path=args.manifest,
                checkpoint=args.checkpoint,
                normalization_path=args.normalization,
                output_dir=args.output_dir,
                split=args.split,
                probability_threshold=args.probability_threshold,
                include_pysteps=not args.no_pysteps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
