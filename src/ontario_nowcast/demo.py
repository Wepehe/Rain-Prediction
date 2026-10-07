"""Deterministic end-to-end smoke demo; never presented as scientific performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation.metrics import categorical_metrics, fractions_skill_score
from .evaluation.onset import first_crossing_minutes, onset_metrics
from .events.detect import detect_event_masks
from .models.baselines import optical_flow_extrapolation, persistence


def synthetic_sequence(frames: int = 36, size: int = 96) -> np.ndarray:
    y, x = np.indices((size, size), dtype=float)
    result = []
    for step in range(frames):
        core_x = 18 + 1.25 * step
        core_y = 58 - 0.45 * step
        amplitude = 10 / (1 + np.exp(-(step - 10) / 2.5))
        primary = amplitude * np.exp(-((x - core_x) ** 2 / 95 + (y - core_y) ** 2 / 55))
        initiation = max(0, step - 18) * 0.8 * np.exp(-((x - 65) ** 2 + (y - 32) ** 2) / 35)
        result.append(primary + initiation)
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence = synthetic_sequence()
    anchor = 17
    truth = sequence[anchor + 1 : anchor + 13]
    persist = persistence(sequence[anchor], len(truth))
    flow = optical_flow_extrapolation(sequence[anchor - 1], sequence[anchor], len(truth))
    events = detect_event_masks(sequence, history_steps=6, horizon_steps=12)
    results = {"warning": "synthetic smoke-test metrics; not scientific results", "thresholds": {}}
    for threshold in (0.1, 1.0, 5.0):
        results["thresholds"][str(threshold)] = {
            "persistence": categorical_metrics(truth, persist, threshold),
            "optical_flow": categorical_metrics(truth, flow, threshold),
            "persistence_fss_9": fractions_skill_score(truth, persist, threshold, 9),
            "optical_flow_fss_9": fractions_skill_score(truth, flow, threshold, 9),
        }
    observed_onset = first_crossing_minutes(truth, 0.1, 6)
    predicted_onset = first_crossing_minutes(flow, 0.1, 6)
    results["optical_flow_onset"] = onset_metrics(observed_onset, predicted_onset)
    results["clear_to_rain_pixel_anchors"] = int(events.clear_to_rain.sum())
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "sequence.npz", rate_mm_hr=sequence)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

