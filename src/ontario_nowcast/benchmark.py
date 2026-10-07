"""Milestone 1.5 benchmark orchestration and report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data.sample import fetch_event
from .events.hard_negatives import mine_fused_file
from .sample_evaluation import evaluate


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _processed_event_path(data_root: Path, event_id: str) -> Path:
    return data_root / "processed" / "events" / f"{event_id}.npz"


def _event_missing_stats(path: Path) -> dict[str, float | int]:
    payload = np.load(path)
    rates = payload["rate_mm_hr"]
    frame_missing = np.mean(~np.isfinite(rates), axis=(1, 2))
    return {
        "frames": int(rates.shape[0]),
        "missing_frames": int(np.sum(frame_missing >= 0.999)),
        "mean_missing_fraction": float(np.mean(frame_missing)),
        "max_missing_fraction": float(np.max(frame_missing)),
    }


def _summarize_event_metrics(report: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "event_id": Path(report["dataset"]).stem,
        "split": event["split"],
        "class": event["class"],
        "type_labels": ", ".join(event.get("type_labels", [])),
        "frames": report["frames"],
        "independent_initiation_events": report.get("independent_initiation_events", 0),
        "apparent_initiation_events": report.get("apparent_initiation_events", 0),
        "anchors_evaluated": report.get("anchors_evaluated", 0),
        "pysteps_status": report.get("pysteps_status", "not_requested"),
    }
    for lead in ("30", "60", "120"):
        lead_metrics = report.get("leads", {}).get(lead, {})
        for model in ("persistence", "optical_flow", "pysteps_extrapolation"):
            threshold_metrics = lead_metrics.get(model, {}).get("1.0")
            if threshold_metrics:
                summary[f"{model}_csi_{lead}min_thr1"] = threshold_metrics["csi"]
                summary[f"{model}_fss18_{lead}min_thr1"] = threshold_metrics[
                    "fss_radius_18km"
                ]
    return summary


def _aggregate_metrics(reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for report in reports:
        for lead, lead_metrics in report.get("leads", {}).items():
            for model, thresholds in lead_metrics.items():
                for threshold, scores in thresholds.items():
                    rows.append(
                        {
                            "lead_minutes": int(lead),
                            "model": model,
                            "threshold_mm_hr": float(threshold),
                            **scores,
                        }
                    )
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    count_columns = ["hits", "misses", "false_alarms", "correct_negatives"]
    grouped = table.groupby(["lead_minutes", "model", "threshold_mm_hr"], as_index=False)
    summed = grouped[count_columns].sum()
    averaged = grouped[
        ["csi", "pod", "far", "precision", "recall", "f1", "ets"]
        + [column for column in table.columns if column.startswith("fss_radius_")]
    ].mean()
    return summed.merge(averaged, on=["lead_minutes", "model", "threshold_mm_hr"])


def _markdown_table(table: pd.DataFrame, columns: list[str], *, limit: int | None = None) -> str:
    if table.empty:
        return "_No rows available._"
    available = [column for column in columns if column in table.columns]
    if not available:
        return "_No requested columns available._"
    subset = table.loc[:, available].head(limit).copy() if limit else table.loc[:, available].copy()
    headers = list(subset.columns)
    rows = []
    for _, row in subset.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            elif isinstance(value, list):
                values.append(", ".join(str(item) for item in value))
            else:
                values.append(str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _load_report_if_reusable(
    event_artifacts: Path,
    *,
    include_pysteps: bool,
    include_steps_ensemble: bool,
    anchor_stride: int,
    max_anchors: int | None,
) -> dict[str, Any] | None:
    report_path = event_artifacts / "baseline_metrics.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    leads = report.get("leads", {})
    lead_metrics = next(iter(leads.values()), {})
    if include_pysteps and "pysteps_extrapolation" not in lead_metrics:
        return None
    if include_steps_ensemble and "pysteps_steps_ensemble_mean" not in lead_metrics:
        return None
    settings = report.get("evaluation_settings", {})
    if int(settings.get("anchor_stride", 1)) != anchor_stride:
        return None
    if settings.get("max_anchors") != max_anchors:
        return None
    return report


def _load_hard_negative_pairs(artifact_root: Path) -> pd.DataFrame:
    path = artifact_root / "hard_negative_pair_matching.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _bool_count(table: pd.DataFrame, column: str) -> int:
    if table.empty or column not in table.columns:
        return 0
    return int(table[column].astype(str).str.lower().eq("true").sum())


def _write_report(
    config: dict[str, Any],
    output: Path,
    event_rows: pd.DataFrame,
    aggregate_rows: pd.DataFrame,
    hard_negative_count: int,
    missing_events: list[str],
    hard_negative_pairs: pd.DataFrame | None = None,
) -> None:
    events = config["events"]
    hard_negative_events = config.get("hard_negative_events", [])
    positive_count = sum(1 for event in events if event["class"].startswith("positive"))
    initiation_count = sum(event["class"] == "positive_initiation" for event in events)
    test_count = sum(event["split"] == "test" for event in events)
    evaluated_count = len(event_rows)
    py_status = sorted(set(event_rows.get("pysteps_status", pd.Series(dtype=str)).dropna()))
    pysteps_note = (
        "PySTEPS is installed in the local environment and the deterministic extrapolation "
        "baseline was included in the current aggregate metrics."
        if "available" in py_status
        else """PySTEPS is wired through `ontario_nowcast.models.baselines`, but the local Windows/Python 3.12
environment could not build the package without Microsoft C++ Build Tools. The recommended
reproducible route is a Python 3.11 conda-forge environment, then rerun:

```powershell
nowcast-run-benchmark --config configs/data/milestone_1_5.yaml --include-pysteps
```"""
    )
    mean_initiations = (
        mean(event_rows["independent_initiation_events"]) if evaluated_count else float("nan")
    )
    total_anchors = (
        int(event_rows["anchors_evaluated"].fillna(0).astype(int).sum())
        if "anchors_evaluated" in event_rows
        else 0
    )
    hard_negative_pairs = hard_negative_pairs if hard_negative_pairs is not None else pd.DataFrame()
    verified_hard_negatives = _bool_count(hard_negative_pairs, "verified_hard_negative")
    strict_dry_negatives = _bool_count(hard_negative_pairs, "strict_dry_window")
    sparse_dry_negatives = _bool_count(hard_negative_pairs, "sparse_dry_window")
    stage_3_authorized = (
        evaluated_count == len(events)
        and not missing_events
        and verified_hard_negatives == len(hard_negative_events)
        and "available" in py_status
    )
    content = f"""# Milestone 1.5 benchmark

This milestone constructs the first southern-Ontario benchmark before learned-model training.
The benchmark manifest currently contains {len(events)} candidate weather-event windows:
{positive_count} positive precipitation windows, {initiation_count} explicit initiation-focused
windows, and {test_count} event-level test windows, plus {len(hard_negative_events)} matched
candidate hard-negative windows. Events are split only at the whole-event level; linked storm
fragments from a test event must not enter training.

## Status

- Evaluated MRMS samples available locally: {evaluated_count}
- Manifest events not yet materialized locally: {len(missing_events)}
- Mean linked initiation objects per evaluated event: {mean_initiations:.1f}
- Verified matched hard-negative windows: {verified_hard_negatives}/{len(hard_negative_events)}
- Sparse-radar hard-negative windows: {sparse_dry_negatives}/{len(hard_negative_events)}
- Strictly all-pixel dry hard-negative windows: {strict_dry_negatives}/{len(hard_negative_events)}
- Conservative hard-negative objects currently mined from fused multimodal data: {hard_negative_count}
- PySTEPS status from this machine: {', '.join(py_status) if py_status else 'not run'}
- Stage 3 radar-only learned-model gate: {'authorized' if stage_3_authorized else 'blocked'}

{pysteps_note}

The wrapper follows the public PySTEPS extrapolation example and API documentation:
https://pysteps.readthedocs.io/en/latest/auto_examples/plot_extrapolation_nowcast.html and
https://pysteps.readthedocs.io/en/stable/generated/pysteps.nowcasts.steps.forecast.html.

## Selected Events

{_markdown_table(pd.DataFrame(events), ['id', 'split', 'class', 'type_labels', 'paired_negative_id'], limit=None)}

## Matched Candidate Hard Negatives

{_markdown_table(pd.DataFrame(hard_negative_events), ['id', 'match_to', 'split', 'type_labels'], limit=None)}

## Evaluated Event Summary

{_markdown_table(event_rows, ['event_id', 'split', 'class', 'frames', 'anchors_evaluated', 'independent_initiation_events', 'apparent_initiation_events', 'pysteps_status'], limit=None)}

## Aggregate Baselines

The table below aggregates currently materialized events only. All baselines use the same anchors,
valid masks, lead times, thresholds, and observations. FSS is reported by physical neighbourhood
radius, not pixel count. The current benchmark pass evaluated {total_anchors} forecast anchors across
the positive windows; per-event anchor counts are shown above so strided PySTEPS runs remain auditable.

{_markdown_table(aggregate_rows.query('threshold_mm_hr == 1.0') if not aggregate_rows.empty else aggregate_rows, ['lead_minutes', 'model', 'threshold_mm_hr', 'csi', 'far', 'fss_radius_18km'], limit=36)}

## Event Independence Logic

The detector first requires a dry radar history, future precipitation within the configured horizon,
and sufficient observed pixels. It applies a southern-Ontario polygon mask, simple morphological
opening/closing, and minimum object size filtering. Components are linked into one object track when
they are close in time and satisfy either bounding-box overlap or centroid-distance criteria. One
representative initiation is kept per linked object, which suppresses lifecycle fragments and nearby
duplicate anchor times.

Apparent initiation is separated from likely advective entry by two conservative flags: objects near
the southern-Ontario polygon boundary and objects adjacent to rain already present at issue time are
marked `advective_entry_like`. These rows are retained for audit but should not be treated as clean
initiation positives.

Known failure modes: the polygon is deliberately approximate, lake-effect bands crossing the mask
edge may be over-flagged as advective entry, split/merge storm behaviour is represented by simple
component links, and no environmental wind vector is yet used to distinguish growth from advection.

## Hard Negatives

Hard negatives are intended to match favourable environments that do not initiate precipitation.
The verifier records radar coverage first, then promotes only sparse-radar windows to GOES/HRRR
pair matching. `strict_dry_window` means no observed pixel in the event crop exceeds 0.1 mm h⁻¹.
`sparse_dry_window` is a pragmatic large-domain criterion that permits tiny radar speckle/edge
areas while rejecting windows with spatially extensive precipitation. Windows rejected by radar are
kept in the table so the negative set is auditable instead of silently curated.

{_markdown_table(hard_negative_pairs, ['positive_event_id', 'negative_event_id', 'verified_hard_negative', 'rejection_reason', 'wet_pixel_fraction', 'max_frame_wet_pixel_fraction', 'heavy_pixel_fraction', 'p99_rate_mm_hr', 'mean_abs_standardized_difference'], limit=None)}

## Stage 3 Gate Decision

Stage 3 is {'authorized' if stage_3_authorized else 'blocked'} for the first radar-only learned
experiment. The gate requires all
18 positive events to be materialized/evaluated, PySTEPS to be included, event-level split
independence to remain intact, and all 18 paired hard negatives to pass radar plus GOES/HRRR
verification. If any of those checks fail, the next action is benchmark repair rather than learned
model training. The Stage 3 split is frozen in
`configs/experiments/stage_3_split_manifest.yaml`; held-out test events are reserved for one final
evaluation after architecture, loss weights, normalization, and stopping rules are selected from
train/dev results.

## Multimodal Features

Retained verified features are GOES C13 brightness temperature, HRRR 2 m temperature, 2 m dew point,
10 m U/V wind, mean sea-level pressure, surface CAPE, and nearby ECCC hourly station observations.
HRRR reprojection now uses projection-aware linear interpolation in EPSG:3978 with nearest fill only
outside the local convex hull. Candidate additions such as CIN, precipitable water, humidity and wind
aloft, vertical velocity, convergence, shear, and lapse-rate variables remain gated on unit,
valid-time, and archive-continuity checks.

## Missing Data

{_markdown_table(event_rows, ['event_id', 'frames', 'missing_frames', 'mean_missing_fraction', 'max_missing_fraction'], limit=None)}

Missing or unmaterialized events: {', '.join(missing_events) if missing_events else 'none'}.

## Visual Checks

Per-event overview plots are written beside each evaluated event under
`artifacts/milestone_1_5/<event_id>/`. Dedicated audit panels are under
`artifacts/milestone_1_5/visual_audit/`:

- `optical_flow_success_late_june_2024.png`
- `initiation_growth_failure_spring_showers_2025.png`
- `dissipation_case_august_2025.png`
- `verified_hard_negative_august_2024.png`

These panels compare issue-time radar, +60 minute observations, optical-flow forecasts, PySTEPS
forecasts, and forecast errors for representative positive cases, plus a sparse-radar verified
negative-window overview.

## Storage Estimate For First Learned Experiment

The first learned experiment should stay radar-only until it beats PySTEPS on this benchmark. A
reasonable starting payload is 18 windows x roughly 8 hours x 6-minute cadence x one 2 km southern
Ontario crop. That is small enough to keep as event tensors plus masks and metadata, well below the
multi-year storage estimates from Milestone 1. Full province-wide continuous storage remains
explicitly out of scope.

## Smallest Learned Experiment After This Milestone

Use a radar-only sequence-to-sequence baseline with 60 minutes of MRMS input and 0-120 minutes of
output on fixed 256 x 256 km tiles sampled from the benchmark objects. Train only on event-level
training windows, tune on dev windows, and report once on held-out test events. It must beat
persistence, Farneback, and PySTEPS deterministic extrapolation at initiation-heavy leads before any
multimodal model is worth training.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")


def run_benchmark(
    config_path: Path,
    data_root: Path,
    artifact_root: Path,
    report_path: Path,
    *,
    fetch_missing: bool = False,
    max_events: int | None = None,
    include_pysteps: bool = False,
    include_steps_ensemble: bool = False,
    materialize_only: bool = False,
    reuse_existing: bool = True,
    anchor_stride: int = 1,
    max_anchors: int | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    events = config["events"][:max_events] if max_events is not None else config["events"]
    event_rows = []
    reports = []
    missing_events = []
    failed_events = []
    for index, event in enumerate(events, start=1):
        path = _processed_event_path(data_root, event["id"])
        if not path.exists() and fetch_missing:
            print(f"[{index}/{len(events)}] fetching {event['id']}", flush=True)
            try:
                path = fetch_event(
                    config_path, event["id"], data_root, config["benchmark"]["cadence_minutes"]
                )
            except Exception as exc:  # noqa: BLE001 - benchmark should continue and report failures
                failed_events.append({"event_id": event["id"], "stage": "fetch", "error": str(exc)})
                missing_events.append(event["id"])
                continue
        if not path.exists():
            missing_events.append(event["id"])
            continue
        if materialize_only:
            event_rows.append(
                {
                    "event_id": event["id"],
                    "split": event["split"],
                    "class": event["class"],
                    "type_labels": ", ".join(event.get("type_labels", [])),
                    **_event_missing_stats(path),
                }
            )
            continue
        event_artifacts = artifact_root / event["id"]
        try:
            print(f"[{index}/{len(events)}] evaluating {event['id']}", flush=True)
            report = (
                _load_report_if_reusable(
                    event_artifacts,
                    include_pysteps=include_pysteps,
                    include_steps_ensemble=include_steps_ensemble,
                    anchor_stride=anchor_stride,
                    max_anchors=max_anchors,
                )
                if reuse_existing
                else None
            )
            if report is None:
                report = evaluate(
                    path,
                    event_artifacts,
                    config["benchmark"]["cadence_minutes"],
                    resolution_km=config["benchmark"]["evaluation_resolution_km"],
                    include_pysteps=include_pysteps,
                    include_steps_ensemble=include_steps_ensemble,
                    anchor_stride=anchor_stride,
                    max_anchors=max_anchors,
                )
            else:
                print(f"[{index}/{len(events)}] reused cached metrics for {event['id']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep other events moving
            failed_events.append({"event_id": event["id"], "stage": "evaluate", "error": str(exc)})
            continue
        reports.append(report)
        event_rows.append({**_summarize_event_metrics(report, event), **_event_missing_stats(path)})

    event_table = pd.DataFrame(event_rows)
    aggregate_table = _aggregate_metrics(reports)
    artifact_root.mkdir(parents=True, exist_ok=True)
    event_table.to_csv(artifact_root / "event_summary.csv", index=False)
    aggregate_table.to_csv(artifact_root / "aggregate_metrics.csv", index=False)
    pd.DataFrame(failed_events).to_csv(artifact_root / "failed_events.csv", index=False)

    fused_path = data_root / "processed" / "fused" / "toronto_2024_07_16.npz"
    hard_negative_count = 0
    if fused_path.exists():
        negatives = mine_fused_file(
            str(fused_path), str(artifact_root / "hard_negative_candidates.csv")
        )
        hard_negative_count = len(negatives)

    if not materialize_only:
        _write_report(
            config,
            report_path,
            event_table,
            aggregate_table,
            hard_negative_count,
            missing_events,
            _load_hard_negative_pairs(artifact_root),
        )
    summary = {
        "positive_events_in_manifest": len(config["events"]),
        "hard_negative_events_in_manifest": len(config.get("hard_negative_events", [])),
        "positive_events_attempted": len(events),
        "events_evaluated": len(event_table),
        "events_missing": len(missing_events),
        "events_failed": len(failed_events),
        "materialize_only": materialize_only,
        "anchor_stride": anchor_stride,
        "max_anchors": max_anchors,
        "hard_negative_candidates": hard_negative_count,
        "report": report_path.as_posix(),
    }
    (artifact_root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/milestone_1_5.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/milestone_1_5"))
    parser.add_argument(
        "--report", type=Path, default=Path("docs/milestone_1_5_benchmark.md")
    )
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--include-pysteps", action="store_true")
    parser.add_argument("--include-steps-ensemble", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--refresh-evaluation", action="store_true")
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--max-anchors", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(
                args.config,
                args.data_root,
                args.artifact_root,
                args.report,
                fetch_missing=args.fetch_missing,
                max_events=args.max_events,
                include_pysteps=args.include_pysteps,
                include_steps_ensemble=args.include_steps_ensemble,
                materialize_only=args.materialize_only,
                reuse_existing=not args.refresh_evaluation,
                anchor_stride=args.anchor_stride,
                max_anchors=args.max_anchors,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
