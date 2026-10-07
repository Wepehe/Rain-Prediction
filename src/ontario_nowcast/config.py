"""Configuration loading with lightweight validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("version") != 1:
        raise ValueError("Only configuration version 1 is supported")
    if config.get("projection") != "EPSG:3978":
        raise ValueError("Milestone 1 requires the documented EPSG:3978 analysis grid")
    return config

