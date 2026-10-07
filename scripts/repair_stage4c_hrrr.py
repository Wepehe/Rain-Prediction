"""Redownload the five HRRR subsets found incomplete by the Stage 4C audit."""

from pathlib import Path

from ontario_nowcast.training.stage4_materialize import (
    HRRR_VARIABLE_MESSAGES,
    _download_hrrr_subset,
    _hrrr_message_ranges,
)

keys = [
    "hrrr.20240403/conus/hrrr.t13z.wrfsfcf03.grib2",
    "hrrr.20240817/conus/hrrr.t17z.wrfsfcf01.grib2",
    "hrrr.20240817/conus/hrrr.t17z.wrfsfcf02.grib2",
    "hrrr.20240817/conus/hrrr.t17z.wrfsfcf03.grib2",
    "hrrr.20240827/conus/hrrr.t17z.wrfsfcf01.grib2",
]
variables = ["TMP_2m", "DPT_2m", "CAPE_surface", "CIN_surface", "PWAT_column"]
selectors = {name: HRRR_VARIABLE_MESSAGES[name][1] for name in variables}
for number, key in enumerate(keys, 1):
    destination = (
        Path("data/raw/nwp/hrrr/stage4") / key.replace("/conus/", "/")
    ).with_suffix(".stage4_subset.grib2")
    ranges = _hrrr_message_ranges(key, selectors)
    _download_hrrr_subset(key, ranges, destination, Path("data/metadata/downloads.jsonl"))
    print(f"repaired {number}/{len(keys)} {key}", flush=True)
