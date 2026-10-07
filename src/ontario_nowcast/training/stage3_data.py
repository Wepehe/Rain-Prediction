"""Radar-only Stage 3 tile manifests and PyTorch datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..sample_evaluation import _downsample_max

CATEGORIES = (
    "initiation_centered",
    "active_precipitation",
    "dissipation",
    "random_weather_event",
    "dry_low_activity",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _event_lookup(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {event["id"]: event for event in benchmark["events"]}


def _merged_event_lookup(experiment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    benchmark = _load_yaml(Path(experiment["benchmark_config"]))
    events = _event_lookup(benchmark)
    for config_path in experiment.get("extra_event_configs", []):
        extra = _load_yaml(Path(config_path))
        for event in extra.get("events", []):
            if event["id"] in events:
                raise ValueError(f"duplicate event id in experiment inputs: {event['id']}")
            events[event["id"]] = event
    return events


def _split_lookup(split_manifest: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    split_keys = (
        ("train", "train_events"),
        ("dev", "dev_events"),
        ("test", "test_events"),
        ("existing_test", "existing_test_events"),
        ("fresh_holdout", "fresh_holdout_events"),
    )
    for split, key in split_keys:
        for event_id in split_manifest.get(key, []):
            if event_id in result:
                raise ValueError(f"event appears in more than one split: {event_id}")
            result[event_id] = split
    return result


def _load_event(event_id: str, data_root: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    payload = np.load(data_root / "processed" / "events" / f"{event_id}.npz")
    return _downsample_max(payload["rate_mm_hr"], 2), pd.to_datetime(payload["times"], utc=True)


def _event_artifact_root(event_id: str, roots: list[Path]) -> Path:
    for root in roots:
        if (root / event_id).exists():
            return root
    return roots[0]


def _tile_bounds(center_y: int, center_x: int, tile_pixels: int, ny: int, nx: int) -> tuple[int, int]:
    if tile_pixels > ny or tile_pixels > nx:
        raise ValueError(f"tile {tile_pixels} does not fit in event crop {(ny, nx)}")
    y0 = int(np.clip(center_y - tile_pixels // 2, 0, ny - tile_pixels))
    x0 = int(np.clip(center_x - tile_pixels // 2, 0, nx - tile_pixels))
    return y0, x0


def _wet_fraction(values: np.ndarray, threshold: float) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite] >= threshold))


def _target_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {category: total * float(weights.get(category, 0.0)) for category in CATEGORIES}
    counts = {category: int(np.floor(value)) for category, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(CATEGORIES, key=lambda category: raw[category] - counts[category], reverse=True)
    for category in order[:remainder]:
        counts[category] += 1
    return counts


def _candidate_centers(ny: int, nx: int, tile_pixels: int, rng: np.random.Generator) -> tuple[int, int]:
    return (
        int(rng.integers(tile_pixels // 2, ny - tile_pixels // 2 + 1)),
        int(rng.integers(tile_pixels // 2, nx - tile_pixels // 2 + 1)),
    )


def _time_index(times: pd.DatetimeIndex, value: str) -> int | None:
    matches = np.where(times == pd.Timestamp(value).tz_convert("UTC"))[0]
    return int(matches[0]) if len(matches) else None


def _initiation_samples(
    event_id: str,
    rates: np.ndarray,
    times: pd.DatetimeIndex,
    event_artifact_root: Path,
    *,
    input_frames: int,
    target_frames: int,
    tile_pixels: int,
    limit: int,
) -> list[dict[str, int | str]]:
    path = event_artifact_root / event_id / "independent_initiation_events.csv"
    if not path.exists() or limit <= 0:
        return []
    catalog = pd.read_csv(path)
    if catalog.empty:
        return []
    rows = []
    ny, nx = rates.shape[1:]
    ranked = catalog.sort_values(["apparent_initiation", "future_max_mm_hr"], ascending=[True, False])
    for _, item in ranked.iterrows():
        anchor = _time_index(times, str(item["representative_anchor_time"]))
        if anchor is None or anchor < input_frames - 1 or anchor + target_frames >= len(times):
            continue
        center_y = round((float(item["y_min"]) + float(item["y_max"])) / 2)
        center_x = round((float(item["x_min"]) + float(item["x_max"])) / 2)
        y0, x0 = _tile_bounds(center_y, center_x, tile_pixels, ny, nx)
        rows.append({"anchor_index": anchor, "y0": y0, "x0": x0, "category": "initiation_centered"})
        if len(rows) >= limit:
            break
    return rows


def _random_category_samples(
    rates: np.ndarray,
    rng: np.random.Generator,
    *,
    input_frames: int,
    target_frames: int,
    tile_pixels: int,
    threshold: float,
    category: str,
    limit: int,
) -> list[dict[str, int | str]]:
    if limit <= 0:
        return []
    rows = []
    ny, nx = rates.shape[1:]
    anchors = np.arange(input_frames - 1, len(rates) - target_frames)
    if len(anchors) == 0:
        return rows
    max_attempts = max(500, limit * 100)
    seen: set[tuple[int, int, int, str]] = set()
    for _ in range(max_attempts):
        anchor = int(rng.choice(anchors))
        center_y, center_x = _candidate_centers(ny, nx, tile_pixels, rng)
        y0, x0 = _tile_bounds(center_y, center_x, tile_pixels, ny, nx)
        key = (anchor, y0, x0, category)
        if key in seen:
            continue
        seen.add(key)
        current = rates[anchor, y0 : y0 + tile_pixels, x0 : x0 + tile_pixels]
        future = rates[anchor + 1 : anchor + target_frames + 1, y0 : y0 + tile_pixels, x0 : x0 + tile_pixels]
        current_wet = _wet_fraction(current, threshold)
        future_wet = _wet_fraction(future, threshold)
        accepts = {
            "active_precipitation": current_wet >= 0.02,
            "dissipation": current_wet >= 0.02 and future_wet <= current_wet * 0.55,
            "random_weather_event": True,
            "dry_low_activity": max(current_wet, future_wet) <= 0.002,
        }
        if accepts[category]:
            rows.append({"anchor_index": anchor, "y0": y0, "x0": x0, "category": category})
        if len(rows) >= limit:
            break
    return rows


def build_sample_manifest(
    experiment_config_path: Path,
    *,
    data_root: Path = Path("data"),
    artifact_root: Path = Path("artifacts/milestone_1_5"),
    output_path: Path = Path("artifacts/stage_3/sample_manifest.csv"),
    include_test: bool = True,
) -> pd.DataFrame:
    """Build a deterministic sample manifest without loading target frames into inputs."""
    experiment = _load_yaml(experiment_config_path)
    split_manifest = _load_yaml(Path(experiment["split_manifest"]))
    split_by_event = _split_lookup(split_manifest)
    benchmark = _load_yaml(Path(experiment["benchmark_config"]))
    events = _merged_event_lookup(experiment)
    cfg_data = experiment["data"]
    cfg_sampling = experiment["sampling"]
    artifact_roots = [artifact_root, *(Path(path) for path in experiment.get("event_artifact_roots", []))]
    input_frames = int(cfg_data["input_frames"])
    target_frames = int(cfg_data["target_frames"])
    tile_pixels = int(cfg_data["tile_pixels"])
    threshold = float(cfg_data["rain_threshold_mm_hr"])
    rng = np.random.default_rng(int(cfg_sampling["seed"]))
    rows: list[dict[str, Any]] = []
    limits = {
        "train": int(cfg_sampling["max_train_samples_per_event"]),
        "dev": int(cfg_sampling["max_dev_samples_per_event"]),
        "test": int(cfg_sampling.get("max_test_samples_per_event", 0)),
        "existing_test": int(
            cfg_sampling.get(
                "max_existing_test_samples_per_event",
                cfg_sampling.get("max_test_samples_per_event", 0),
            )
        ),
        "fresh_holdout": int(cfg_sampling.get("max_fresh_holdout_samples_per_event", 0)),
    }
    def append_rows(
        *,
        event_id: str,
        split: str,
        event_class: str,
        samples: list[dict[str, int | str]],
        times: pd.DatetimeIndex,
    ) -> None:
        for sample_index, sample in enumerate(samples):
            anchor = int(sample["anchor_index"])
            rows.append(
                {
                    "sample_id": f"{event_id}_{sample_index:04d}",
                    "split": split,
                    "event_id": event_id,
                    "event_class": event_class,
                    "category": sample["category"],
                    "input_start_index": anchor - input_frames + 1,
                    "input_end_index": anchor,
                    "target_start_index": anchor + 1,
                    "target_end_index": anchor + target_frames,
                    "input_start_time_utc": times[anchor - input_frames + 1].isoformat(),
                    "issue_time_utc": times[anchor].isoformat(),
                    "target_start_time_utc": times[anchor + 1].isoformat(),
                    "target_end_time_utc": times[anchor + target_frames].isoformat(),
                    "y0": sample["y0"],
                    "x0": sample["x0"],
                    "tile_pixels": tile_pixels,
                    "grid_resolution_km": cfg_data["grid_resolution_km"],
                    "tile_size_km": cfg_data["tile_size_km"],
                    "rain_threshold_mm_hr": threshold,
                }
            )

    for event_id, split in split_by_event.items():
        if split in {"test", "existing_test", "fresh_holdout"} and not include_test:
            continue
        print(f"[stage3-data] sampling {split} event {event_id}", flush=True)
        event = events[event_id]
        rates, times = _load_event(event_id, data_root)
        counts = _target_counts(limits[split], cfg_sampling["category_weights"])
        samples = _initiation_samples(
            event_id,
            rates,
            times,
            _event_artifact_root(event_id, artifact_roots),
            input_frames=input_frames,
            target_frames=target_frames,
            tile_pixels=tile_pixels,
            limit=counts["initiation_centered"],
        )
        for category in CATEGORIES[1:]:
            samples.extend(
                _random_category_samples(
                    rates,
                    rng,
                    input_frames=input_frames,
                    target_frames=target_frames,
                    tile_pixels=tile_pixels,
                    threshold=threshold,
                    category=category,
                    limit=counts[category],
                )
            )
        print(
            f"[stage3-data] sampled {event_id}: requested={limits[split]} actual={len(samples)}",
            flush=True,
        )
        append_rows(event_id=event_id, split=split, event_class=event["class"], samples=samples, times=times)
    negative_limit = int(cfg_sampling.get("hard_negative_dry_samples_per_event", 0))
    for event in benchmark.get("hard_negative_events", []):
        split = event["split"]
        if split == "test" and "existing_test_events" in split_manifest:
            split = "existing_test"
        if split in {"test", "existing_test", "fresh_holdout"} and not include_test:
            continue
        print(f"[stage3-data] sampling hard-negative {split} event {event['id']}", flush=True)
        rates, times = _load_event(event["id"], data_root)
        samples = _random_category_samples(
            rates,
            rng,
            input_frames=input_frames,
            target_frames=target_frames,
            tile_pixels=tile_pixels,
            threshold=threshold,
            category="dry_low_activity",
            limit=negative_limit,
        )
        append_rows(
            event_id=event["id"],
            split=split,
            event_class=event["class"],
            samples=samples,
            times=times,
        )
        print(
            f"[stage3-data] sampled hard-negative {event['id']}: "
            f"requested={negative_limit} actual={len(samples)}",
            flush=True,
        )
    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    summary = {
        "experiment_config": experiment_config_path.as_posix(),
        "samples": len(table),
        "by_split": table.groupby("split").size().to_dict() if not table.empty else {},
        "by_category": table.groupby("category").size().to_dict() if not table.empty else {},
    }
    (output_path.parent / "sample_manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return table


@dataclass(frozen=True)
class Stage3Example:
    inputs: Any
    target_occurrence: Any
    target_intensity: Any
    target_mask: Any
    metadata: dict[str, Any]


class Stage3RadarDataset:
    """PyTorch dataset returning radar history + validity mask and future targets."""

    def __init__(
        self,
        manifest: pd.DataFrame | Path,
        *,
        data_root: Path = Path("data"),
        split: str | None = None,
        normalization: dict[str, float] | None = None,
        preload: bool = False,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the Stage 3 learned-model dependency: torch") from exc

        self.torch = torch
        table = pd.read_csv(manifest) if isinstance(manifest, Path) else manifest.copy()
        if split is not None:
            table = table[table["split"] == split].copy()
        self.table = table.reset_index(drop=True)
        self.data_root = data_root
        self.normalization = normalization or {"mean": 0.0, "std": 1.0}
        self._cache: dict[str, np.ndarray] = {}
        if preload:
            for event_id in sorted(self.table["event_id"].astype(str).unique()):
                print(f"[stage3-data] preloading event {event_id}", flush=True)
                self._rates(event_id)

    def __len__(self) -> int:
        return len(self.table)

    def _rates(self, event_id: str) -> np.ndarray:
        if event_id not in self._cache:
            rates, _ = _load_event(event_id, self.data_root)
            self._cache[event_id] = rates
        return self._cache[event_id]

    def __getitem__(self, index: int) -> Stage3Example:
        row = self.table.iloc[index]
        rates = self._rates(str(row["event_id"]))
        y0, x0, size = int(row["y0"]), int(row["x0"]), int(row["tile_pixels"])
        history = rates[
            int(row["input_start_index"]) : int(row["input_end_index"]) + 1,
            y0 : y0 + size,
            x0 : x0 + size,
        ]
        target = rates[
            int(row["target_start_index"]) : int(row["target_end_index"]) + 1,
            y0 : y0 + size,
            x0 : x0 + size,
        ]
        history_mask = np.isfinite(history).astype(np.float32)
        target_mask = np.isfinite(target).astype(np.float32)
        history_log = np.where(history_mask > 0, np.log1p(np.nan_to_num(history)), 0.0)
        target_log = np.where(target_mask > 0, np.log1p(np.nan_to_num(target)), 0.0)
        mean = float(self.normalization["mean"])
        std = max(float(self.normalization["std"]), 1e-6)
        history_norm = (history_log - mean) / std
        inputs = np.stack([history_norm, history_mask], axis=1).astype(np.float32)
        occurrence = ((target >= float(row["rain_threshold_mm_hr"])) & np.isfinite(target)).astype(
            np.float32
        )
        return Stage3Example(
            inputs=self.torch.from_numpy(inputs),
            target_occurrence=self.torch.from_numpy(occurrence),
            target_intensity=self.torch.from_numpy(target_log.astype(np.float32)),
            target_mask=self.torch.from_numpy(target_mask),
            metadata=row.to_dict(),
        )


def collate_stage3(examples: list[Stage3Example]) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    import torch

    return {
        "inputs": torch.stack([example.inputs for example in examples]),
        "target_occurrence": torch.stack([example.target_occurrence for example in examples]),
        "target_intensity": torch.stack([example.target_intensity for example in examples]),
        "target_mask": torch.stack([example.target_mask for example in examples]),
        "metadata": [example.metadata for example in examples],
    }


def compute_train_normalization(
    manifest: pd.DataFrame | Path,
    *,
    data_root: Path = Path("data"),
) -> dict[str, float]:
    table = pd.read_csv(manifest) if isinstance(manifest, Path) else manifest.copy()
    count = 0
    total = 0.0
    total_sq = 0.0
    for event_id, event_rows in table[table["split"] == "train"].groupby("event_id"):
        rates, _ = _load_event(str(event_id), data_root)
        for row in event_rows.itertuples(index=False):
            history = rates[
                int(row.input_start_index) : int(row.input_end_index) + 1,
                int(row.y0) : int(row.y0) + int(row.tile_pixels),
                int(row.x0) : int(row.x0) + int(row.tile_pixels),
            ]
            finite = np.isfinite(history)
            if not np.any(finite):
                continue
            values = np.log1p(history[finite])
            count += int(values.size)
            total += float(values.sum())
            total_sq += float(np.square(values).sum())
    if count == 0:
        raise ValueError("no finite training inputs found for normalization")
    mean = total / count
    variance = max(total_sq / count - mean**2, 0.0)
    return {"mean": float(mean), "std": float(np.sqrt(variance))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/stage_3_radar_only.yaml")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/milestone_1_5"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage_3/sample_manifest.csv"))
    parser.add_argument("--exclude-test", action="store_true")
    args = parser.parse_args()
    table = build_sample_manifest(
        args.config,
        data_root=args.data_root,
        artifact_root=args.artifact_root,
        output_path=args.output,
        include_test=not args.exclude_test,
    )
    print(table.groupby(["split", "category"]).size().to_string())


if __name__ == "__main__":
    main()
