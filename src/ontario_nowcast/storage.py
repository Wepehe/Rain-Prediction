"""Transparent, assumption-driven storage estimates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config

MINUTES_PER_YEAR = 365.25 * 24 * 60


def estimate_storage(config: dict[str, Any], years: float, domain: str = "research") -> dict[str, float]:
    """Estimate processed and raw storage without pretending source compression is exact."""
    if years <= 0:
        raise ValueError("years must be positive")
    domain_cfg = config["domains"][domain]
    resolution = float(domain_cfg["resolution_km"])
    nx = math.ceil(float(domain_cfg["approximate_width_km"]) / resolution)
    ny = math.ceil(float(domain_cfg["approximate_height_km"]) / resolution)
    storage_cfg = config["storage"]
    channels = sum(
        int(storage_cfg[name])
        for name in (
            "radar_channels",
            "satellite_channels",
            "nwp_channels",
            "station_feature_channels",
        )
    )
    frames = MINUTES_PER_YEAR * years / float(storage_cfg["processed_cadence_minutes"])
    uncompressed = nx * ny * channels * frames * int(storage_cfg["bytes_per_stored_value"])
    processed = uncompressed / float(storage_cfg["compression_ratio"])
    raw = processed * float(storage_cfg["raw_overhead_factor"])
    return {
        "years": years,
        "nx": nx,
        "ny": ny,
        "channels": channels,
        "frames": frames,
        "processed_tib": processed / 2**40,
        "raw_plus_processed_tib": (processed + raw) / 2**40,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/milestone_1.yaml"))
    parser.add_argument("--domain", default="research")
    parser.add_argument("--years", type=float, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps([estimate_storage(config, y, args.domain) for y in args.years], indent=2))


if __name__ == "__main__":
    main()

