import numpy as np
import pandas as pd
import yaml

from ontario_nowcast.training.stage3_data import (
    Stage3RadarDataset,
    compute_train_normalization,
)


def test_stage3_frozen_split_manifest_is_disjoint() -> None:
    with open("configs/experiments/stage_3_split_manifest.yaml", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    train = set(manifest["train_events"])
    dev = set(manifest["dev_events"])
    test = set(manifest["test_events"])
    assert train.isdisjoint(dev)
    assert train.isdisjoint(test)
    assert dev.isdisjoint(test)
    assert test == {
        "late_june_organized_storms_2024_06_22",
        "august_severe_convection_2024_08_27",
        "lake_effect_precip_2024_11_29",
        "july_organized_convection_2025_07_13",
    }


def test_stage3_manifest_has_no_future_input_leakage() -> None:
    table = pd.read_csv("artifacts/stage_3/sample_manifest.csv")
    assert (table["input_end_index"] < table["target_start_index"]).all()
    assert pd.to_datetime(table["issue_time_utc"], utc=True).lt(
        pd.to_datetime(table["target_start_time_utc"], utc=True)
    ).all()


def _write_event(tmp_path, event_id: str, value: float) -> None:
    event_dir = tmp_path / "processed" / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    rates = np.full((32, 256, 256), value, dtype=np.float32)
    rates[0, :2, :2] = np.nan
    np.savez_compressed(
        event_dir / f"{event_id}.npz",
        rate_mm_hr=rates,
        times=np.array([f"2024-01-01T00:{index:02d}:00+00:00" for index in range(32)]),
        latitude=np.linspace(45, 40, 256),
        longitude=np.linspace(-85, -75, 256),
    )


def test_stage3_normalization_uses_train_split_only(tmp_path) -> None:
    _write_event(tmp_path, "train_event", 3.0)
    _write_event(tmp_path, "test_event", 100.0)
    manifest = pd.DataFrame(
        [
            {
                "split": "train",
                "event_id": "train_event",
                "input_start_index": 0,
                "input_end_index": 9,
                "target_start_index": 10,
                "target_end_index": 29,
                "y0": 0,
                "x0": 0,
                "tile_pixels": 128,
                "rain_threshold_mm_hr": 0.1,
            },
            {
                "split": "test",
                "event_id": "test_event",
                "input_start_index": 0,
                "input_end_index": 9,
                "target_start_index": 10,
                "target_end_index": 29,
                "y0": 0,
                "x0": 0,
                "tile_pixels": 128,
                "rain_threshold_mm_hr": 0.1,
            },
        ]
    )
    stats = compute_train_normalization(manifest, data_root=tmp_path)
    assert np.isclose(stats["mean"], np.log1p(3.0))


def test_stage3_dataset_preserves_missing_mask(tmp_path) -> None:
    _write_event(tmp_path, "train_event", 1.0)
    manifest = pd.DataFrame(
        [
            {
                "split": "train",
                "event_id": "train_event",
                "input_start_index": 0,
                "input_end_index": 9,
                "target_start_index": 10,
                "target_end_index": 29,
                "y0": 0,
                "x0": 0,
                "tile_pixels": 128,
                "rain_threshold_mm_hr": 0.1,
            }
        ]
    )
    dataset = Stage3RadarDataset(manifest, data_root=tmp_path, normalization={"mean": 0.0, "std": 1.0})
    example = dataset[0]
    assert example.inputs.shape == (10, 2, 128, 128)
    assert example.inputs[0, 1, 0, 0].item() == 0.0
    assert example.inputs[0, 0, 0, 0].item() == 0.0
