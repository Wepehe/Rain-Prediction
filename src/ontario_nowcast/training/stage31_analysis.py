"""Stage 3.1 head diagnostics and DEV-only probability threshold analysis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..evaluation.metrics import brier_score
from ..models.radar_convlstm import build_radar_model, expected_rate_mm_hr
from .stage3_data import Stage3RadarDataset, collate_stage3
from .stage3_eval import _observed_targets, _probability_for_threshold


@dataclass
class DistributionAccumulator:
    """Streaming moments plus a bounded sample for quantiles."""

    max_sample: int = 200_000
    seed: int = 20260904
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = np.inf
    maximum: float = -np.inf
    sample: list[np.ndarray] = field(default_factory=list)

    def update(self, values: np.ndarray) -> None:
        finite_values = values[np.isfinite(values)].astype(np.float32, copy=False).reshape(-1)
        if finite_values.size == 0:
            return
        self.count += int(finite_values.size)
        self.total += float(finite_values.sum(dtype=np.float64))
        self.total_sq += float(np.square(finite_values, dtype=np.float64).sum())
        self.minimum = min(self.minimum, float(finite_values.min()))
        self.maximum = max(self.maximum, float(finite_values.max()))
        rng = np.random.default_rng(self.seed + self.count)
        take = min(finite_values.size, 5_000)
        if finite_values.size > take:
            finite_values = finite_values[rng.choice(finite_values.size, size=take, replace=False)]
        self.sample.append(finite_values.copy())
        total_sample = sum(part.size for part in self.sample)
        if total_sample > self.max_sample:
            combined = np.concatenate(self.sample)
            keep = rng.choice(combined.size, size=self.max_sample, replace=False)
            self.sample = [combined[keep]]

    def as_row(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "p01": np.nan,
                "p05": np.nan,
                "p10": np.nan,
                "p25": np.nan,
                "p50": np.nan,
                "p75": np.nan,
                "p90": np.nan,
                "p95": np.nan,
                "p99": np.nan,
                "max": np.nan,
            }
        sample = np.concatenate(self.sample) if self.sample else np.asarray([], dtype=np.float32)
        quantiles = np.nanpercentile(sample, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean**2, 0.0)
        return {
            "count": self.count,
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "min": self.minimum,
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p10": float(quantiles[2]),
            "p25": float(quantiles[3]),
            "p50": float(quantiles[4]),
            "p75": float(quantiles[5]),
            "p90": float(quantiles[6]),
            "p95": float(quantiles[7]),
            "p99": float(quantiles[8]),
            "max": self.maximum,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_model(config: dict[str, Any], checkpoint: Path, device: Any) -> Any:
    import torch

    model = build_radar_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def _pixel_groups(observed: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "dry_pixels_observed_rate_lt_0p1": observed < 0.1,
        "light_rain_pixels_observed_0p1_to_1": (observed >= 0.1) & (observed < 1.0),
        "moderate_heavy_pixels_observed_ge_1": observed >= 1.0,
    }


def _update_distributions(
    accumulators: dict[tuple[str, str, str, str], DistributionAccumulator],
    *,
    split: str,
    group_type: str,
    group_name: str,
    mask: np.ndarray,
    variables: dict[str, np.ndarray],
) -> None:
    valid_mask = mask & np.isfinite(variables["observed_rate_mm_hr"])
    if not np.any(valid_mask):
        return
    for variable_name, values in variables.items():
        accumulators[(split, group_type, group_name, variable_name)].update(values[valid_mask])


def _threshold_rows(
    probability: np.ndarray,
    observed: np.ndarray,
    probability_thresholds: np.ndarray,
    rain_threshold_mm_hr: float,
) -> list[dict[str, float | int]]:
    valid = np.isfinite(probability) & np.isfinite(observed)
    observed_occurrence = observed[valid] >= rain_threshold_mm_hr
    probs = probability[valid]
    rows = []
    for threshold in probability_thresholds:
        predicted = probs >= threshold
        hits = int(np.sum(predicted & observed_occurrence))
        misses = int(np.sum(~predicted & observed_occurrence))
        false_alarms = int(np.sum(predicted & ~observed_occurrence))
        correct_negatives = int(np.sum(~predicted & ~observed_occurrence))
        csi_den = hits + misses + false_alarms
        pod_den = hits + misses
        far_den = hits + false_alarms
        f1_den = 2 * hits + false_alarms + misses
        rows.append(
            {
                "probability_threshold": float(threshold),
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_negatives": correct_negatives,
                "csi": hits / csi_den if csi_den else np.nan,
                "pod": hits / pod_den if pod_den else np.nan,
                "far": false_alarms / far_den if far_den else np.nan,
                "precision": hits / far_den if far_den else np.nan,
                "recall": hits / pod_den if pod_den else np.nan,
                "f1": 2 * hits / f1_den if f1_den else np.nan,
            }
        )
    return rows


def _reliability_rows(probability: np.ndarray, observed: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    valid = np.isfinite(probability) & np.isfinite(observed)
    probs = probability[valid]
    obs = observed[valid] >= threshold
    bins = np.minimum((probs * 10).astype(int), 9)
    rows = []
    for index in range(10):
        keep = bins == index
        count = int(np.sum(keep))
        rows.append(
            {
                "bin_min": index / 10,
                "bin_max": (index + 1) / 10,
                "count": count,
                "mean_forecast_probability": float(np.mean(probs[keep])) if count else np.nan,
                "observed_frequency": float(np.mean(obs[keep])) if count else np.nan,
            }
        )
    return rows


def _plot_threshold_sweep(table: pd.DataFrame, destination: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(table["recall"], table["precision"], marker="o")
    axes[0].set_xlabel("Recall / POD")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("DEV precision-recall: P(rate > 0.1)")
    for metric in ["csi", "pod", "far", "f1"]:
        axes[1].plot(table["probability_threshold"], table[metric], marker="o", label=metric)
    axes[1].set_xlabel("Probability threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("DEV scores vs probability threshold")
    axes[1].legend()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def analyze_stage31_heads(
    config_path: Path,
    *,
    manifest_path: Path,
    checkpoint: Path,
    normalization_path: Path,
    output_dir: Path,
    splits: list[str],
    max_samples: int | None = None,
) -> dict[str, Any]:
    import torch

    config = _load_yaml(config_path)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(config, checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulators: dict[tuple[str, str, str, str], DistributionAccumulator] = defaultdict(
        DistributionAccumulator
    )
    dev_probability_parts = []
    dev_observed_parts = []
    sample_counts: dict[str, int] = {}
    rain_threshold = float(config["data"]["rain_threshold_mm_hr"])
    for split in splits:
        dataset = Stage3RadarDataset(manifest_path, split=split, normalization=normalization)
        if max_samples is not None:
            dataset.table = dataset.table.head(max_samples).reset_index(drop=True)
        sample_counts[split] = len(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_stage3)
        for index, batch in enumerate(loader, start=1):
            metadata = batch["metadata"][0]
            model_batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            with torch.no_grad():
                occurrence_logits, intensity_raw = model(model_batch["inputs"])
                probability_raw = torch.sigmoid(occurrence_logits).detach().cpu().numpy()[0]
                probability = _probability_for_threshold(
                    probability_raw, config["model"], rain_threshold
                )
                expected_rate = expected_rate_mm_hr(occurrence_logits, intensity_raw).detach().cpu().numpy()[0]
                conditional_rate = (
                    torch.expm1(torch.nn.functional.softplus(intensity_raw)).detach().cpu().numpy()[0]
                )
            observed = _observed_targets(batch)[0]
            variables = {
                "predicted_p_rate_ge_0p1": probability,
                "predicted_conditional_rate_mm_hr": conditional_rate,
                "combined_expected_rate_mm_hr": expected_rate,
                "observed_rate_mm_hr": observed,
            }
            for group_name, mask in _pixel_groups(observed).items():
                _update_distributions(
                    accumulators,
                    split=split,
                    group_type="observed_pixel_class",
                    group_name=group_name,
                    mask=mask,
                    variables=variables,
                )
            category = str(metadata["category"])
            if category in {"initiation_centered", "dissipation"}:
                _update_distributions(
                    accumulators,
                    split=split,
                    group_type="sample_category",
                    group_name=category,
                    mask=np.isfinite(observed),
                    variables=variables,
                )
            if split == "dev":
                dev_probability_parts.append(probability.reshape(-1))
                dev_observed_parts.append(observed.reshape(-1))
            if index == 1 or index % 25 == 0:
                print(f"[stage31-analysis] {split}: {index}/{len(dataset)} samples", flush=True)
    distribution_rows = [
        {
            "split": split,
            "group_type": group_type,
            "group": group_name,
            "variable": variable_name,
            **accumulator.as_row(),
        }
        for (split, group_type, group_name, variable_name), accumulator in sorted(
            accumulators.items()
        )
    ]
    distributions_path = output_dir / "head_distributions.csv"
    pd.DataFrame(distribution_rows).to_csv(distributions_path, index=False)
    probability_thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    threshold_path = None
    reliability_path = None
    selected_threshold = None
    dev_brier = None
    if dev_probability_parts:
        dev_probability = np.concatenate(dev_probability_parts)
        dev_observed = np.concatenate(dev_observed_parts)
        threshold_table = pd.DataFrame(
            _threshold_rows(dev_probability, dev_observed, probability_thresholds, rain_threshold)
        )
        threshold_path = output_dir / "dev_probability_threshold_sweep.csv"
        threshold_table.to_csv(threshold_path, index=False)
        reliability = pd.DataFrame(_reliability_rows(dev_probability, dev_observed, rain_threshold))
        reliability_path = output_dir / "dev_probability_reliability.csv"
        reliability.to_csv(reliability_path, index=False)
        selected = threshold_table.sort_values(["f1", "csi"], ascending=False).iloc[0]
        selected_threshold = float(selected["probability_threshold"])
        dev_brier = brier_score((dev_observed >= rain_threshold).astype(float), dev_probability)
        _plot_threshold_sweep(threshold_table, output_dir / "dev_probability_threshold_sweep.png")
    manifest = pd.read_csv(manifest_path)
    split_category = (
        manifest.groupby(["split", "category"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
    )
    split_totals = manifest.groupby("split").size().rename("split_samples")
    split_category = split_category.merge(split_totals, on="split")
    split_category["fraction"] = split_category["samples"] / split_category["split_samples"]
    split_category_path = output_dir / "split_category_fractions.csv"
    split_category.to_csv(split_category_path, index=False)
    summary = {
        "config": config_path.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "splits": splits,
        "samples": sample_counts,
        "quantity_definitions": {
            "predicted_p_rate_ge_0p1": "occurrence-head probability for rate >= 0.1 mm/h",
            "predicted_conditional_rate_mm_hr": "exp(softplus intensity head) - 1",
            "combined_expected_rate_mm_hr": "P(rate >= 0.1) times conditional intensity",
            "observed_rate_mm_hr": "MRMS target precipitation rate",
            "dev_threshold_sweep": "categorical rain from P(rate >= 0.1) >= probability threshold",
        },
        "dev_brier_score_rate_ge_0p1": dev_brier,
        "dev_selected_operational_occurrence_probability_threshold": selected_threshold,
        "outputs": {
            "head_distributions": distributions_path.as_posix(),
            "dev_probability_threshold_sweep": threshold_path.as_posix() if threshold_path else None,
            "dev_probability_reliability": reliability_path.as_posix() if reliability_path else None,
            "split_category_fractions": split_category_path.as_posix(),
        },
    }
    (output_dir / "head_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_3_1/head_analysis"))
    parser.add_argument("--splits", nargs="+", default=["dev", "existing_test"])
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_stage31_heads(
                args.config,
                manifest_path=args.manifest,
                checkpoint=args.checkpoint,
                normalization_path=args.normalization,
                output_dir=args.output_dir,
                splits=args.splits,
                max_samples=args.max_samples,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
