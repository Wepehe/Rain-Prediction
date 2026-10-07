"""Evaluate persistence/optical-flow baselines and mine initiation candidates in a sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation.metrics import categorical_metrics, fractions_skill_score_km
from .events.catalog import build_initiation_catalog
from .events.detect import detect_event_masks
from .events.tracking import build_independent_initiation_catalog
from .models.baselines import (
    optical_flow_extrapolation,
    persistence,
    pysteps_deterministic_extrapolation,
    pysteps_steps_ensemble_mean,
)


def _downsample_max(values: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return values
    nt, ny, nx = values.shape
    ny2, nx2 = ny // factor, nx // factor
    trimmed = values[:, : ny2 * factor, : nx2 * factor]
    blocks = trimmed.reshape(nt, ny2, factor, nx2, factor)
    finite = np.isfinite(blocks)
    pooled = np.max(np.where(finite, blocks, -np.inf), axis=(2, 4))
    pooled[~np.any(finite, axis=(2, 4))] = np.nan
    return pooled


def evaluate(
    path: Path,
    output_dir: Path,
    interval_minutes: int = 6,
    *,
    resolution_km: float = 2.0,
    include_pysteps: bool = False,
    include_steps_ensemble: bool = False,
    anchor_stride: int = 1,
    max_anchors: int | None = None,
) -> dict:
    if anchor_stride < 1:
        raise ValueError("anchor_stride must be at least 1")
    if max_anchors is not None and max_anchors < 1:
        raise ValueError("max_anchors must be positive when provided")
    payload = np.load(path)
    rates_native = payload["rate_mm_hr"]
    factor = 2
    rates = _downsample_max(rates_native, factor)
    times = pd.to_datetime(payload["times"], utc=True)
    latitudes = payload["latitude"][: rates.shape[1] * factor : factor]
    longitudes = payload["longitude"][: rates.shape[2] * factor : factor]
    history_steps = max(2, 30 // interval_minutes)
    horizon_steps = min(120 // interval_minutes, len(rates) - history_steps)
    event_masks = detect_event_masks(
        rates,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        rain_threshold=0.1,
        strong_threshold=5.0,
    )
    catalog = build_initiation_catalog(
        event_masks.clear_to_rain,
        rates,
        times,
        latitudes,
        longitudes,
        horizon_steps=horizon_steps,
        minimum_pixels=9,
    )
    independent_catalog = build_independent_initiation_catalog(
        rates,
        times,
        latitudes,
        longitudes,
        interval_minutes=interval_minutes,
        resolution_km=resolution_km,
        history_minutes=30,
        horizon_minutes=horizon_steps * interval_minutes,
        rain_threshold=0.1,
        strong_threshold=5.0,
    )

    truth_by_lead: dict[int, list[np.ndarray]] = {
        lead: [] for lead in (1, 2, 3, 5, 8, 10, 15, 20)
    }
    persistence_by_lead = {lead: [] for lead in truth_by_lead}
    flow_by_lead = {lead: [] for lead in truth_by_lead}
    pysteps_by_lead = {lead: [] for lead in truth_by_lead}
    steps_mean_by_lead = {lead: [] for lead in truth_by_lead}
    pysteps_status = "not_requested"
    max_lead = max(truth_by_lead)
    anchors = list(range(2, len(rates) - max_lead, anchor_stride))
    if max_anchors is not None and len(anchors) > max_anchors:
        positions = np.linspace(0, len(anchors) - 1, max_anchors, dtype=int)
        anchors = [anchors[position] for position in positions]
    for anchor in anchors:
        history = rates[anchor - 2 : anchor + 1]
        input_missing_fraction = np.mean(~np.isfinite(history))
        if input_missing_fraction > 0.1:
            continue
        flow = optical_flow_extrapolation(rates[anchor - 1], rates[anchor], max_lead)
        persisted = persistence(rates[anchor], max_lead)
        pysteps_forecast = None
        steps_mean = None
        if include_pysteps:
            try:
                pysteps_forecast = pysteps_deterministic_extrapolation(history, max_lead)
                pysteps_status = "available"
                if include_steps_ensemble:
                    steps_mean = pysteps_steps_ensemble_mean(
                        history,
                        max_lead,
                        km_per_pixel=resolution_km,
                        timestep_minutes=interval_minutes,
                    )
            except RuntimeError as exc:
                pysteps_status = f"unavailable: {exc}"
                include_pysteps = False
        for lead, truth_frames in truth_by_lead.items():
            truth_frames.append(rates[anchor + lead])
            persistence_by_lead[lead].append(persisted[lead - 1])
            flow_by_lead[lead].append(flow[lead - 1])
            if pysteps_forecast is not None:
                pysteps_by_lead[lead].append(pysteps_forecast[lead - 1])
            if steps_mean is not None:
                steps_mean_by_lead[lead].append(steps_mean[lead - 1])

    report: dict = {
        "dataset": path.as_posix(),
        "warning": "Single-event pilot results; not a held-out model comparison.",
        "frames": len(rates),
        "native_shape": list(rates_native.shape),
        "evaluation_shape": list(rates.shape),
        "initiation_candidates": len(catalog),
        "independent_initiation_events": len(independent_catalog),
        "apparent_initiation_events": int(independent_catalog.get("apparent_initiation", []).sum())
        if len(independent_catalog)
        else 0,
        "pysteps_status": pysteps_status,
        "anchors_evaluated": len(anchors),
        "evaluation_settings": {
            "anchor_stride": anchor_stride,
            "max_anchors": max_anchors,
        },
        "fss_neighbourhood_radii_km": [6.0, 18.0, 36.0],
        "leads": {},
    }
    for lead, truth_frames in truth_by_lead.items():
        if not truth_frames:
            continue
        observed = np.stack(truth_frames)
        forecasts = {
            "persistence": np.stack(persistence_by_lead[lead]),
            "optical_flow": np.stack(flow_by_lead[lead]),
        }
        if pysteps_by_lead[lead]:
            forecasts["pysteps_extrapolation"] = np.stack(pysteps_by_lead[lead])
        if steps_mean_by_lead[lead]:
            forecasts["pysteps_steps_ensemble_mean"] = np.stack(steps_mean_by_lead[lead])
        lead_report = {}
        for model, forecast in forecasts.items():
            lead_report[model] = {}
            for threshold in (0.1, 1.0, 5.0):
                scores = categorical_metrics(observed, forecast, threshold)
                for radius_km in (6.0, 18.0, 36.0):
                    key = f"fss_radius_{radius_km:g}km"
                    scores[key] = fractions_skill_score_km(
                        observed, forecast, threshold, radius_km, resolution_km
                    )
                lead_report[model][str(threshold)] = scores
        report["leads"][str(lead * interval_minutes)] = lead_report

    output_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_dir / "initiation_candidates.csv", index=False)
    independent_catalog.to_csv(output_dir / "independent_initiation_events.csv", index=False)
    (output_dir / "baseline_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot_event(rates, times, output_dir / "event_overview.png")
    return report


def _plot_event(rates: np.ndarray, times: pd.DatetimeIndex, destination: Path) -> None:
    import matplotlib.pyplot as plt

    positions = np.linspace(0, len(rates) - 1, 6, dtype=int)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    image = None
    for axis, position in zip(axes.flat, positions, strict=True):
        image = axis.imshow(np.log1p(rates[position]), vmin=0, vmax=np.log1p(25), cmap="turbo")
        axis.set_title(times[position].strftime("%Y-%m-%d %H:%M UTC"))
        axis.set_axis_off()
    assert image is not None
    figure.colorbar(image, ax=axes, label="log(1 + precipitation rate [mm h⁻¹])", shrink=0.8)
    figure.suptitle("NOAA MRMS event sample (native lat/lon crop, 2× max-pooled)")
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/milestone_1"))
    parser.add_argument("--interval-minutes", type=int, default=6)
    parser.add_argument("--resolution-km", type=float, default=2.0)
    parser.add_argument("--include-pysteps", action="store_true")
    parser.add_argument("--include-steps-ensemble", action="store_true")
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--max-anchors", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.sample,
                args.output_dir,
                args.interval_minutes,
                resolution_km=args.resolution_km,
                include_pysteps=args.include_pysteps,
                include_steps_ensemble=args.include_steps_ensemble,
                anchor_stride=args.anchor_stride,
                max_anchors=args.max_anchors,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
