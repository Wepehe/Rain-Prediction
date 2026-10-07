from pathlib import Path

from ontario_nowcast.config import load_config
from ontario_nowcast.storage import estimate_storage


def test_storage_scales_linearly_by_year() -> None:
    config = load_config(Path("configs/data/milestone_1.yaml"))
    one = estimate_storage(config, 1)
    three = estimate_storage(config, 3)
    assert three["processed_tib"] == 3 * one["processed_tib"]
    assert one["nx"] == 800
    assert one["ny"] == 950

