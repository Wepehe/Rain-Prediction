"""Visual sanity panels for Stage 4 materialized tensors."""

from __future__ import annotations

import argparse
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PANEL_EVENT_SELECTION = {
    "clean_convective_initiation": "stage4_dev_initiation_2026_06_18",
    "organized_storm": "late_june_organized_storms_2024_06_22",
    "stratiform": "september_stratiform_2024_09_24",
    "dissipation": "stage4_dev_dissipation_2026_05_20",
    "matched_favourable_dry_negative": "stage4_dev_dry_favourable_2026_07_10",
}


def _safe_image(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def plot_radar_goes_panel(tensor_path: Path, destination: Path, *, panel_label: str) -> dict[str, Any]:
    """Create one synchronized radar+GOES panel from a materialized tensor."""
    import matplotlib.pyplot as plt

    payload = np.load(tensor_path)
    radar = payload["radar_rate_mm_hr"]
    target = payload["target_rate_mm_hr"]
    goes = payload["goes"]
    channel_names = [str(value) for value in payload["satellite_channel_names"]]
    cooling_30_index = channel_names.index("ABI_C13_cooling_tendency_30min")
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fields = [
        (
            _safe_image(np.log1p(radar[0])),
            "Radar history: oldest",
            "turbo",
            0.0,
            float(np.log1p(25.0)),
            "log1p mm h-1",
        ),
        (
            _safe_image(np.log1p(radar[-1])),
            "Radar at issue time",
            "turbo",
            0.0,
            float(np.log1p(25.0)),
            "log1p mm h-1",
        ),
        (
            _safe_image(goes[0, 0]),
            "GOES C13 latest usable",
            "gray_r",
            210.0,
            300.0,
            "K",
        ),
        (
            _safe_image(goes[cooling_30_index, 0]),
            "C13 cooling, 30 min",
            "RdBu_r",
            -30.0,
            30.0,
            "K; positive = cooling",
        ),
        (
            _safe_image(np.log1p(target[9])),
            "Observed future +60 min",
            "turbo",
            0.0,
            float(np.log1p(25.0)),
            "log1p mm h-1",
        ),
        (
            _safe_image(np.log1p(target[-1])),
            "Observed future +120 min",
            "turbo",
            0.0,
            float(np.log1p(25.0)),
            "log1p mm h-1",
        ),
    ]
    for axis, (values, title, colour_map, minimum, maximum, label) in zip(
        axes.flat, fields, strict=True
    ):
        image = axis.imshow(values, origin="upper", cmap=colour_map, vmin=minimum, vmax=maximum)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=0.75, label=label)
    issue_time = str(payload["forecast_issue_time_utc"])
    figure.suptitle(f"{panel_label}: {Path(tensor_path).stem} issue {issue_time}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return {
        "panel_label": panel_label,
        "tensor_path": tensor_path.as_posix(),
        "panel_path": destination.as_posix(),
        "forecast_issue_time_utc": issue_time,
        "radar_shape": "x".join(str(value) for value in radar.shape),
        "goes_shape": "x".join(str(value) for value in goes.shape),
        "target_shape": "x".join(str(value) for value in target.shape),
        "radar_missing_fraction": float(np.mean(payload["radar_valid_mask"] <= 0)),
        "goes_missing_fraction": float(np.mean(payload["goes_valid_mask"] <= 0)),
        "target_missing_fraction": float(np.mean(payload["target_valid_mask"] <= 0)),
        "cooling_sign_convention": str(payload["cooling_sign_convention"]),
    }


def build_radar_goes_visual_sanity_panels(
    tensor_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create representative B-input visual panels without model inference."""
    manifest = pd.read_csv(tensor_manifest)
    rows = []
    for label, event_id in PANEL_EVENT_SELECTION.items():
        candidates = manifest[manifest["event_id"].astype(str).eq(event_id)]
        if candidates.empty:
            rows.append(
                {
                    "panel_label": label,
                    "event_id": event_id,
                    "status": "missing_tensor_for_requested_event",
                }
            )
            continue
        record = candidates.sort_values("forecast_issue_time_utc").iloc[len(candidates) // 2]
        panel_path = output_dir / f"{label}__{record['anchor_id']}.png"
        panel = plot_radar_goes_panel(
            Path(str(record["tensor_path"])),
            panel_path,
            panel_label=label,
        )
        rows.append({"event_id": event_id, "status": "created", **panel})
    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "stage4_radar_goes_visual_panels.csv", index=False)
    summary = {
        "created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "tensor_manifest": tensor_manifest.as_posix(),
        "output_dir": output_dir.as_posix(),
        "requested_panels": len(PANEL_EVENT_SELECTION),
        "created_panels": int(table["status"].eq("created").sum()) if not table.empty else 0,
        "complete": bool(
            not table.empty and table["status"].eq("created").all() and len(table) == len(PANEL_EVENT_SELECTION)
        ),
        "includes_nwp_fields": False,
        "training_started": False,
    }
    (output_dir / "stage4_radar_goes_visual_sanity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor-manifest",
        type=Path,
        default=Path("artifacts/stage_4/materialization/stage4_radar_goes_tensor_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4/visual_sanity"))
    args = parser.parse_args()
    print(
        json.dumps(
            build_radar_goes_visual_sanity_panels(args.tensor_manifest, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
