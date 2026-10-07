"""DEV lifecycle and probability-threshold analysis for Stage 4B."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..evaluation.metrics import brier_score
from ..evaluation.onset import first_crossing_minutes, onset_metrics
from ..models.baselines import (
    optical_flow_extrapolation,
    persistence,
    pysteps_deterministic_extrapolation,
)
from ..models.radar_convlstm import build_radar_model, expected_rate_mm_hr
from ..models.radar_goes import build_stage4b_model
from .stage4b_train import HOLDOUT_SPLITS, _load_npz

RAIN_THRESHOLD = 0.1
INTERVAL_MINUTES = 6
DEFAULT_PROBABILITY_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _radar_inputs(payload: dict[str, np.ndarray], normalization: dict[str, Any]) -> np.ndarray:
    radar = payload["radar"]
    mask = payload["radar_mask"]
    radar_log = np.where(mask > 0, np.log1p(np.nan_to_num(radar, nan=0.0)), 0.0)
    mean = float(normalization["radar"]["mean"])
    std = max(float(normalization["radar"]["std"]), 1e-6)
    return np.stack([(radar_log - mean) / std, mask], axis=1).astype(np.float32)


def _goes_inputs(
    payload: dict[str, np.ndarray],
    normalization: dict[str, Any],
    *,
    channel_indices: list[int],
) -> np.ndarray:
    goes = payload["goes"][channel_indices]
    mask = payload["goes_mask"][channel_indices]
    mean = np.asarray(normalization["goes"]["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(normalization["goes"]["std"], dtype=np.float32), 1e-6)
    goes_norm = (np.nan_to_num(goes, nan=0.0) - mean[:, None, None, None]) / (
        std[:, None, None, None]
    )
    values_time_major = np.moveaxis(goes_norm, 1, 0)
    mask_time_major = np.moveaxis(mask, 1, 0)
    return np.concatenate([values_time_major, mask_time_major], axis=1).astype(np.float32)


def _event_probability_from_sequence(probability: np.ndarray) -> np.ndarray:
    return np.nanmax(probability, axis=0)


def _append_lifecycle_row(
    rows: list[dict[str, Any]],
    *,
    lifecycle: str,
    model: str,
    prediction_type: str,
    probability_threshold: float | None,
    observed_minutes: list[np.ndarray],
    predicted_minutes: list[np.ndarray],
    observed_event_probabilities: list[np.ndarray],
    predicted_event_probabilities: list[np.ndarray],
    sample_count: int,
) -> None:
    if not observed_minutes:
        return
    observed = np.concatenate([value.reshape(-1) for value in observed_minutes])
    predicted = np.concatenate([value.reshape(-1) for value in predicted_minutes])
    obs_prob = np.concatenate([value.reshape(-1) for value in observed_event_probabilities])
    pred_prob = np.concatenate([value.reshape(-1) for value in predicted_event_probabilities])
    metrics = onset_metrics(observed, predicted)
    observed_events = int(metrics["observed_events"])
    predicted_events = int(metrics["predicted_events"])
    paired_events = int(metrics["paired_events"])
    false_events = int(metrics["false_events"])
    missed_events = int(metrics["missed_events"])
    precision = paired_events / predicted_events if predicted_events else float("nan")
    recall = paired_events / observed_events if observed_events else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else float("nan")
    )
    row = {
        "lifecycle": lifecycle,
        "model": model,
        "prediction_type": prediction_type,
        "probability_threshold": probability_threshold,
        "samples": sample_count,
        **metrics,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_event_fraction_of_predictions": false_events / max(false_events + paired_events, 1),
        "miss_fraction_of_observed": missed_events / observed_events if observed_events else float("nan"),
        "brier_score": brier_score((obs_prob > 0).astype(float), pred_prob),
    }
    for window_minutes in (30, 60, 90, 120):
        observed_in_window = np.isfinite(observed) & (observed <= window_minutes)
        predicted_in_window = np.isfinite(predicted) & (predicted <= window_minutes)
        hits = int(np.sum(observed_in_window & predicted_in_window))
        misses = int(np.sum(observed_in_window & ~predicted_in_window))
        false_alarms = int(np.sum(~observed_in_window & predicted_in_window))
        row[f"detection_within_{window_minutes}min_recall"] = (
            hits / (hits + misses) if hits + misses else float("nan")
        )
        row[f"detection_within_{window_minutes}min_precision"] = (
            hits / (hits + false_alarms) if hits + false_alarms else float("nan")
        )
    rows.append(row)


def _model_outputs_for_tensor(
    *,
    payload: dict[str, np.ndarray],
    models: dict[str, Any],
    metadata: dict[str, Any],
    device: Any,
    include_pysteps: bool,
) -> dict[str, dict[str, np.ndarray]]:
    import torch

    outputs: dict[str, dict[str, np.ndarray]] = {}
    radar_input = torch.from_numpy(
        _radar_inputs(payload, metadata["B2"]["training_metadata"]["normalization"])[None]
    ).to(device)
    with torch.no_grad():
        radar_logits, radar_intensity = models["stage31_radar"](radar_input)
        outputs["stage31_radar_expected"] = {
            "rate": expected_rate_mm_hr(radar_logits, radar_intensity).detach().cpu().numpy()[0],
            "probability": torch.sigmoid(radar_logits).detach().cpu().numpy()[0, :, 0],
        }
        if "A_plus_radar" in models:
            aplus_logits, aplus_intensity = models["A_plus_radar"](radar_input)
            outputs["A_plus_radar_expected"] = {
                "rate": expected_rate_mm_hr(aplus_logits, aplus_intensity)
                .detach()
                .cpu()
                .numpy()[0],
                "probability": torch.sigmoid(aplus_logits).detach().cpu().numpy()[0, :, 0],
            }
    for name in ("B1", "B2"):
        cfg = metadata[name]["experiment_config"]
        channel_indices = [int(value) for value in cfg["data"]["goes_channel_indices"]]
        goes_input = torch.from_numpy(
            _goes_inputs(
                payload,
                metadata[name]["training_metadata"]["normalization"],
                channel_indices=channel_indices,
            )[None]
        ).to(device)
        with torch.no_grad():
            logits, intensity = models[name](radar_input, goes_input)
            outputs[f"{name}_expected"] = {
                "rate": expected_rate_mm_hr(logits, intensity).detach().cpu().numpy()[0],
                "probability": torch.sigmoid(logits).detach().cpu().numpy()[0, :, 0],
            }
    history = payload["radar"]
    outputs["persistence"] = {
        "rate": persistence(history[-1], history.shape[0] * 2),
        "probability": None,
    }
    outputs["optical_flow"] = {
        "rate": optical_flow_extrapolation(history[-2], history[-1], history.shape[0] * 2),
        "probability": None,
    }
    if include_pysteps:
        outputs["pysteps"] = {
            "rate": pysteps_deterministic_extrapolation(history[-3:], history.shape[0] * 2),
            "probability": None,
        }
    return outputs


def run_stage4b_lifecycle_dev(
    *,
    output_dir: Path,
    tensor_manifest: Path = Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
    b1_config: Path = Path("configs/experiments/stage_4b_b1_raw_c13.yaml"),
    b2_config: Path = Path("configs/experiments/stage_4b_b2_c13_cooling.yaml"),
    b1_checkpoint: Path = Path("artifacts/stage_4b/b1_raw_c13/model_best.pt"),
    b2_checkpoint: Path = Path("artifacts/stage_4b/b2_c13_cooling/model_best.pt"),
    b1_metadata: Path = Path("artifacts/stage_4b/b1_raw_c13/training_metadata.json"),
    b2_metadata: Path = Path("artifacts/stage_4b/b2_c13_cooling/training_metadata.json"),
    radar_config: Path = Path("configs/experiments/stage_3_1_radar_control.yaml"),
    radar_checkpoint: Path = Path(
        "artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/model_best.pt"
    ),
    aplus_config: Path = Path("configs/experiments/stage_4b_a_plus_radar_capacity.yaml"),
    aplus_checkpoint: Path = Path("artifacts/stage_4b/a_plus_radar/model_best.pt"),
    include_pysteps: bool = True,
) -> dict[str, Any]:
    import torch

    manifest = pd.read_csv(tensor_manifest)
    dev = manifest[manifest["split"].astype(str).isin({"dev", "stage4_dev_repair"})].copy()
    if dev["split"].astype(str).isin(HOLDOUT_SPLITS).any():
        raise ValueError("DEV lifecycle analysis must not include final initiation holdout")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = {
        "B1": {
            "experiment_config": _load_yaml(b1_config),
            "training_metadata": json.loads(b1_metadata.read_text(encoding="utf-8")),
        },
        "B2": {
            "experiment_config": _load_yaml(b2_config),
            "training_metadata": json.loads(b2_metadata.read_text(encoding="utf-8")),
        },
    }
    radar_model = build_radar_model(_load_yaml(radar_config)).to(device)
    radar_model.load_state_dict(torch.load(radar_checkpoint, map_location=device))
    models = {
        "stage31_radar": radar_model.eval(),
        "B1": build_stage4b_model(metadata["B1"]["experiment_config"]).to(device).eval(),
        "B2": build_stage4b_model(metadata["B2"]["experiment_config"]).to(device).eval(),
    }
    if aplus_config.exists() and aplus_checkpoint.exists():
        aplus_cfg = _load_yaml(aplus_config)
        aplus_model = build_radar_model({"model": aplus_cfg["model"], "data": aplus_cfg["data"]}).to(
            device
        )
        aplus_model.load_state_dict(torch.load(aplus_checkpoint, map_location=device))
        models["A_plus_radar"] = aplus_model.eval()
    models["B1"].load_state_dict(torch.load(b1_checkpoint, map_location=device))
    models["B2"].load_state_dict(torch.load(b2_checkpoint, map_location=device))

    grouped: dict[tuple[str, str, str, float | None], dict[str, Any]] = defaultdict(
        lambda: {
            "observed_minutes": [],
            "predicted_minutes": [],
            "observed_event_probabilities": [],
            "predicted_event_probabilities": [],
            "sample_count": 0,
        }
    )
    for index, row in enumerate(dev.itertuples(index=False), start=1):
        payload = _load_npz(Path(str(row.tensor_path)))
        observed = payload["target"]
        latest = payload["radar"][-1]
        event_class = str(row.event_class)
        event_id = str(row.event_id)
        outputs = _model_outputs_for_tensor(
            payload=payload,
            models=models,
            metadata=metadata,
            device=device,
            include_pysteps=include_pysteps,
        )
        if event_class == "positive_initiation":
            lifecycle = "initiation"
            mask = np.isfinite(latest) & (latest < RAIN_THRESHOLD)
            observed_minutes = first_crossing_minutes(observed, RAIN_THRESHOLD, INTERVAL_MINUTES)
            observed_event_probability = np.isfinite(observed_minutes).astype(float)
            for model_name, output in outputs.items():
                rate = output["rate"]
                predicted_expected = first_crossing_minutes(rate, RAIN_THRESHOLD, INTERVAL_MINUTES)
                key = (lifecycle, model_name, "expected_rate_field", None)
                grouped[key]["observed_minutes"].append(observed_minutes[mask])
                grouped[key]["predicted_minutes"].append(predicted_expected[mask])
                grouped[key]["observed_event_probabilities"].append(observed_event_probability[mask])
                grouped[key]["predicted_event_probabilities"].append(
                    (np.isfinite(predicted_expected[mask])).astype(float)
                )
                grouped[key]["sample_count"] += 1
                if output["probability"] is not None:
                    probability = output["probability"]
                    event_probability = _event_probability_from_sequence(probability)
                    for threshold in DEFAULT_PROBABILITY_THRESHOLDS:
                        predicted_probability = first_crossing_minutes(
                            probability, threshold, INTERVAL_MINUTES
                        )
                        key = (lifecycle, model_name, "probability_head", float(threshold))
                        grouped[key]["observed_minutes"].append(observed_minutes[mask])
                        grouped[key]["predicted_minutes"].append(predicted_probability[mask])
                        grouped[key]["observed_event_probabilities"].append(
                            observed_event_probability[mask]
                        )
                        grouped[key]["predicted_event_probabilities"].append(
                            event_probability[mask]
                        )
                        grouped[key]["sample_count"] += 1
        if event_id == "stage4_dev_dissipation_2026_05_20":
            lifecycle = "dissipation"
            mask = np.isfinite(latest) & (latest >= RAIN_THRESHOLD)
            observed_minutes = first_crossing_minutes(
                observed, RAIN_THRESHOLD, INTERVAL_MINUTES, above=False
            )
            observed_event_probability = np.isfinite(observed_minutes).astype(float)
            for model_name, output in outputs.items():
                rate = output["rate"]
                predicted_expected = first_crossing_minutes(
                    rate, RAIN_THRESHOLD, INTERVAL_MINUTES, above=False
                )
                key = (lifecycle, model_name, "expected_rate_field", None)
                grouped[key]["observed_minutes"].append(observed_minutes[mask])
                grouped[key]["predicted_minutes"].append(predicted_expected[mask])
                grouped[key]["observed_event_probabilities"].append(observed_event_probability[mask])
                grouped[key]["predicted_event_probabilities"].append(
                    (np.isfinite(predicted_expected[mask])).astype(float)
                )
                grouped[key]["sample_count"] += 1
                if output["probability"] is not None:
                    probability = 1.0 - output["probability"]
                    event_probability = _event_probability_from_sequence(probability)
                    for threshold in DEFAULT_PROBABILITY_THRESHOLDS:
                        predicted_probability = first_crossing_minutes(
                            probability, threshold, INTERVAL_MINUTES
                        )
                        key = (lifecycle, model_name, "probability_head", float(threshold))
                        grouped[key]["observed_minutes"].append(observed_minutes[mask])
                        grouped[key]["predicted_minutes"].append(predicted_probability[mask])
                        grouped[key]["observed_event_probabilities"].append(
                            observed_event_probability[mask]
                        )
                        grouped[key]["predicted_event_probabilities"].append(
                            event_probability[mask]
                        )
                        grouped[key]["sample_count"] += 1
        if index == 1 or index % 5 == 0:
            print(f"[stage4b-lifecycle] {index}/{len(dev)} DEV tensors", flush=True)

    rows = []
    for (lifecycle, model_name, prediction_type, threshold), values in sorted(grouped.items()):
        _append_lifecycle_row(
            rows,
            lifecycle=lifecycle,
            model=model_name,
            prediction_type=prediction_type,
            probability_threshold=threshold,
            observed_minutes=values["observed_minutes"],
            predicted_minutes=values["predicted_minutes"],
            observed_event_probabilities=values["observed_event_probabilities"],
            predicted_event_probabilities=values["predicted_event_probabilities"],
            sample_count=int(values["sample_count"]),
        )
    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "dev_lifecycle_metrics.csv"
    table.to_csv(metrics_path, index=False)
    threshold_candidates = table[
        (table["lifecycle"] == "initiation")
        & (table["prediction_type"] == "probability_head")
        & (
            table["model"].isin(
                ["A_plus_radar_expected", "B1_expected", "B2_expected", "stage31_radar_expected"]
            )
        )
    ].copy()
    if not threshold_candidates.empty:
        threshold_candidates["selection_score"] = (
            threshold_candidates["f1"].fillna(0)
            + threshold_candidates["recall"].fillna(0)
            + (1 - threshold_candidates["false_event_fraction_of_predictions"].fillna(1))
            + (1 - threshold_candidates["brier_score"].fillna(1))
        )
    threshold_path = output_dir / "dev_probability_policy_sweep.csv"
    threshold_candidates.to_csv(threshold_path, index=False)
    best_policy = (
        threshold_candidates.sort_values(
            ["model", "selection_score", "f1", "brier_score"],
            ascending=[True, False, False, True],
        )
        .groupby("model")
        .head(1)
        .to_dict("records")
        if not threshold_candidates.empty
        else []
    )
    summary = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "samples": len(dev),
        "include_pysteps": include_pysteps,
        "final_holdout_scoring": "not_evaluated",
        "metrics": metrics_path.as_posix(),
        "probability_policy_sweep": threshold_path.as_posix(),
        "best_policy_by_model": best_policy,
    }
    (output_dir / "dev_lifecycle_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4b/lifecycle"))
    parser.add_argument(
        "--tensor-manifest",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
    )
    parser.add_argument("--no-pysteps", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage4b_lifecycle_dev(
                output_dir=args.output_dir,
                tensor_manifest=args.tensor_manifest,
                include_pysteps=not args.no_pysteps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
