"""Post-hoc Stage 3 calibration and qualitative diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..evaluation.metrics import categorical_metrics
from ..models.baselines import pysteps_deterministic_extrapolation
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


def _sample_score(observed: np.ndarray, forecast: np.ndarray, threshold: float = 0.1) -> dict[str, float]:
    metrics = categorical_metrics(observed, forecast, threshold)
    return {
        "csi": metrics["csi"],
        "far": metrics["far"],
        "pod": metrics["pod"],
        "obs_wet_fraction": float(np.nanmean(observed >= threshold)),
        "pred_wet_fraction": float(np.nanmean(forecast >= threshold)),
    }


def _update_reliability(
    bins: dict[int, dict[str, float]], probabilities: np.ndarray, observed: np.ndarray
) -> None:
    valid = np.isfinite(probabilities) & np.isfinite(observed)
    probs = probabilities[valid]
    obs = observed[valid]
    indices = np.minimum((probs * 10).astype(int), 9)
    for index in range(10):
        keep = indices == index
        if not np.any(keep):
            continue
        bins[index]["count"] += int(np.sum(keep))
        bins[index]["probability_sum"] += float(np.sum(probs[keep]))
        bins[index]["observed_sum"] += float(np.sum(obs[keep]))


def _reliability_table(bins: dict[int, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for index in range(10):
        count = bins[index]["count"]
        rows.append(
            {
                "bin_min": index / 10,
                "bin_max": (index + 1) / 10,
                "count": int(count),
                "mean_forecast_probability": bins[index]["probability_sum"] / count
                if count
                else np.nan,
                "observed_frequency": bins[index]["observed_sum"] / count if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_reliability(table: pd.DataFrame, destination: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="perfect")
    axis.plot(
        table["mean_forecast_probability"],
        table["observed_frequency"],
        marker="o",
        label="learned occurrence",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Forecast probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title("Stage 3 occurrence reliability, threshold 0.1 mm h⁻¹")
    axis.legend()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _plot_panel(case: dict[str, Any], destination: Path) -> None:
    import matplotlib.pyplot as plt

    observed = case["observed"]
    learned = case["learned"]
    pysteps = case["pysteps"]
    probability = case["probability"]
    issue = case["issue"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [
        (issue, "issue radar", "turbo", 0, np.log1p(25)),
        (observed, "+60 min observed", "turbo", 0, np.log1p(25)),
        (learned, "+60 min learned expected", "turbo", 0, np.log1p(25)),
        (pysteps, "+60 min PySTEPS", "turbo", 0, np.log1p(25)),
        (probability, "+60 min learned P(rain)", "viridis", 0, 1),
        (learned - observed, "learned error", "coolwarm", -10, 10),
    ]
    image = None
    for axis, (values, title, cmap, vmin, vmax) in zip(axes.flat, panels, strict=True):
        plot_values = np.log1p(values) if "radar" in title or "observed" in title or "expected" in title or "PySTEPS" in title else values
        image = axis.imshow(plot_values, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_axis_off()
    figure.colorbar(image, ax=axes, shrink=0.75)
    figure.suptitle(case["title"])
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def run_diagnostics(
    config_path: Path,
    *,
    manifest_path: Path = Path("artifacts/stage_3/sample_manifest.csv"),
    checkpoint: Path = Path("artifacts/stage_3/model_best.pt"),
    normalization_path: Path = Path("artifacts/stage_3/normalization.json"),
    output_dir: Path = Path("artifacts/stage_3/diagnostics"),
    split: str = "test",
) -> dict[str, Any]:
    import torch

    config = _load_yaml(config_path)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    dataset = Stage3RadarDataset(manifest_path, split=split, normalization=normalization)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_stage3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(config, checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    reliability_bins = defaultdict(lambda: {"count": 0, "probability_sum": 0.0, "observed_sum": 0.0})
    category_rows = []
    cases: dict[str, dict[str, Any]] = {}
    lead_index = 9
    lead_minutes = 60
    for index, batch in enumerate(loader, start=1):
        metadata = batch["metadata"][0]
        category = str(metadata["category"])
        model_batch = {
            key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
        }
        with torch.no_grad():
            occurrence_logits, intensity_raw = model(model_batch["inputs"])
            learned = expected_rate_mm_hr(occurrence_logits, intensity_raw).detach().cpu().numpy()[0]
            probability_raw = torch.sigmoid(occurrence_logits).detach().cpu().numpy()[0]
            probability = _probability_for_threshold(
                probability_raw, config["model"], float(config["data"]["rain_threshold_mm_hr"])
            )
        observed = _observed_targets(batch)[0]
        history = _recover_history(batch, normalization)[0]
        pysteps = pysteps_deterministic_extrapolation(history[-3:], int(config["data"]["target_frames"]))
        _update_reliability(
            reliability_bins,
            probability.reshape(-1),
            (observed >= float(config["data"]["rain_threshold_mm_hr"])).astype(float).reshape(-1),
        )
        learned_score = _sample_score(observed[lead_index], learned[lead_index])
        pysteps_score = _sample_score(observed[lead_index], pysteps[lead_index])
        category_rows.append(
            {
                "sample_id": metadata["sample_id"],
                "event_id": metadata["event_id"],
                "category": category,
                "lead_minutes": lead_minutes,
                **{f"learned_{key}": value for key, value in learned_score.items()},
                **{f"pysteps_{key}": value for key, value in pysteps_score.items()},
                "learned_minus_pysteps_csi": learned_score["csi"] - pysteps_score["csi"],
            }
        )
        case = {
            "title": f"{metadata['sample_id']} ({category})",
            "observed": observed[lead_index],
            "learned": learned[lead_index],
            "pysteps": pysteps[lead_index],
            "probability": probability[lead_index],
            "issue": history[-1],
        }
        delta = learned_score["csi"] - pysteps_score["csi"]
        if "learned_relative_best" not in cases or delta > cases["learned_relative_best"]["delta"]:
            cases["learned_relative_best"] = {**case, "delta": delta}
        if "pysteps_clear_win" not in cases or delta < cases["pysteps_clear_win"]["delta"]:
            cases["pysteps_clear_win"] = {**case, "delta": delta}
        false_load = learned_score["pred_wet_fraction"] - learned_score["obs_wet_fraction"]
        if "learned_hallucination" not in cases or false_load > cases["learned_hallucination"]["delta"]:
            cases["learned_hallucination"] = {**case, "delta": false_load}
        if category == "initiation_centered":
            both = learned_score["csi"] + pysteps_score["csi"]
            if "both_fail_initiation" not in cases or both < cases["both_fail_initiation"]["delta"]:
                cases["both_fail_initiation"] = {**case, "delta": both}
        if category == "dissipation" and (
            "dissipation_audit" not in cases or delta > cases["dissipation_audit"]["delta"]
        ):
            cases["dissipation_audit"] = {**case, "delta": delta}
        if index == 1 or index % 25 == 0:
            print(f"[stage3-diagnostics] {split}: {index}/{len(dataset)} samples", flush=True)
    category_table = pd.DataFrame(category_rows)
    category_metrics = (
        category_table.groupby("category", as_index=False)
        .agg(
            samples=("sample_id", "count"),
            learned_csi=("learned_csi", "mean"),
            pysteps_csi=("pysteps_csi", "mean"),
            learned_far=("learned_far", "mean"),
            pysteps_far=("pysteps_far", "mean"),
            learned_pred_wet_fraction=("learned_pred_wet_fraction", "mean"),
            observed_wet_fraction=("learned_obs_wet_fraction", "mean"),
        )
        .sort_values("category")
    )
    reliability = _reliability_table(reliability_bins)
    category_table.to_csv(output_dir / f"{split}_sample_diagnostics.csv", index=False)
    category_metrics.to_csv(output_dir / f"{split}_category_metrics.csv", index=False)
    reliability.to_csv(output_dir / f"{split}_reliability.csv", index=False)
    _plot_reliability(reliability, output_dir / f"{split}_reliability.png")
    for name, case in cases.items():
        _plot_panel(case, output_dir / f"{split}_{name}.png")
    summary = {
        "split": split,
        "samples": len(dataset),
        "category_metrics": (output_dir / f"{split}_category_metrics.csv").as_posix(),
        "reliability": (output_dir / f"{split}_reliability.csv").as_posix(),
        "panels": sorted(path.as_posix() for path in output_dir.glob(f"{split}_*.png")),
    }
    (output_dir / f"{split}_diagnostics_summary.json").write_text(
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_3/diagnostics"))
    parser.add_argument(
        "--split",
        choices=["dev", "test", "existing_test", "fresh_holdout"],
        default="test",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_diagnostics(
                args.config,
                manifest_path=args.manifest,
                checkpoint=args.checkpoint,
                normalization_path=args.normalization,
                output_dir=args.output_dir,
                split=args.split,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
