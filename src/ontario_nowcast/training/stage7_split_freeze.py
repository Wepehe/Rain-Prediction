"""Declare new independent candidate periods before any future gate development."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path("artifacts/stage_7/new_splits")
CANADA_2020 = (
    "https://www.canada.ca/en/environment-climate-change/services/top-ten-weather-stories/2020.html"
)
CANADA_2021 = (
    "https://www.canada.ca/en/environment-climate-change/services/top-ten-weather-stories/2021.html"
)
CANADA_2022 = (
    "https://www.canada.ca/en/environment-climate-change/services/top-ten-weather-stories/2022.html"
)


def _row(identifier, split, start, end, label, source):
    return {
        "event_id": identifier,
        "split": split,
        "start_utc": start,
        "end_utc": end,
        "bbox_wgs84": "[-84.8, 41.5, -75.5, 46.5]",
        "candidate_regime": label,
        "source": source,
        "status": "candidate_requires_radar_event_mining_and_availability_gate",
        "used_by_stages_3_to_7": False,
    }


def run() -> dict:
    # Date windows are hypotheses/candidates, not labels. Radar-only mining must confirm objects.
    development = [
        _row(
            "stage7_dev_jun10_2020_convection",
            "new_gate_train",
            "2020-06-10T12:00:00Z",
            "2020-06-11T04:00:00Z",
            "summer convection",
            CANADA_2020,
        ),
        _row(
            "stage7_dev_jul09_2020_heavy_rain",
            "new_gate_train",
            "2020-07-09T12:00:00Z",
            "2020-07-10T06:00:00Z",
            "tropical moisture/heavy rain",
            CANADA_2020,
        ),
        _row(
            "stage7_dev_aug01_2020_storms",
            "new_gate_dev",
            "2020-08-01T12:00:00Z",
            "2020-08-03T06:00:00Z",
            "organized storms/tornadoes",
            CANADA_2020,
        ),
        _row(
            "stage7_dev_jun21_2021_frontal_convection",
            "new_gate_train",
            "2021-06-21T12:00:00Z",
            "2021-06-22T04:00:00Z",
            "frontal convection",
            CANADA_2021,
        ),
        _row(
            "stage7_dev_jul15_2021_barrie_outbreak",
            "new_gate_dev",
            "2021-07-15T12:00:00Z",
            "2021-07-16T04:00:00Z",
            "isolated/supercell initiation",
            CANADA_2021,
        ),
        _row(
            "stage7_dev_sep22_2021_rain",
            "new_gate_train",
            "2021-09-22T12:00:00Z",
            "2021-09-24T06:00:00Z",
            "prolonged rain",
            CANADA_2021,
        ),
        _row(
            "stage7_dev_may21_2022_derecho",
            "new_gate_dev",
            "2022-05-21T12:00:00Z",
            "2022-05-22T04:00:00Z",
            "fast organized convection",
            CANADA_2022,
        ),
        _row(
            "stage7_dev_aug29_2022_wet_windy",
            "new_gate_train",
            "2022-08-29T12:00:00Z",
            "2022-08-31T06:00:00Z",
            "late-summer frontal rain",
            CANADA_2022,
        ),
    ]
    final = [
        _row(
            "stage7_final_sep21_2018_ottawa_tornadoes",
            "new_final_holdout",
            "2018-09-21T12:00:00Z",
            "2018-09-22T04:00:00Z",
            "tornado outbreak",
            "https://www.canada.ca/en/environment-climate-change/news/2018/09/updated-statement-by-environment-and-climate-change-canada-six-tornados-now-confirmed-in-eastern-ontario-and-quebec-on-september-21.html",
        ),
        _row(
            "stage7_final_jun02_2019_convection",
            "new_final_holdout",
            "2019-06-02T12:00:00Z",
            "2019-06-03T04:00:00Z",
            "early-summer convection candidate",
            "radar-mined candidate",
        ),
        _row(
            "stage7_final_jul20_2019_storms",
            "new_final_holdout",
            "2019-07-20T12:00:00Z",
            "2019-07-21T04:00:00Z",
            "midsummer convection candidate",
            "radar-mined candidate",
        ),
        _row(
            "stage7_final_sep11_2019_frontal",
            "new_final_holdout",
            "2019-09-11T12:00:00Z",
            "2019-09-12T06:00:00Z",
            "late-summer frontal candidate",
            "radar-mined candidate",
        ),
        _row(
            "stage7_final_jun30_2020_tornado_warning",
            "new_final_holdout",
            "2020-06-30T12:00:00Z",
            "2020-07-01T04:00:00Z",
            "severe convection candidate",
            "https://www.canada.ca/en/services/environment/weather/severeweather/severe-thunderstorms-tornadoes.html",
        ),
        _row(
            "stage7_final_jun16_2022_cold_front",
            "new_final_holdout",
            "2022-06-16T12:00:00Z",
            "2022-06-17T04:00:00Z",
            "cold-front convection candidate",
            CANADA_2022,
        ),
    ]
    ROOT.mkdir(parents=True, exist_ok=True)
    dev = pd.DataFrame(development)
    holdout = pd.DataFrame(final)
    if set(dev.event_id) & set(holdout.event_id):
        raise RuntimeError("new development and final candidate periods overlap")
    dev_path, final_path = (
        ROOT / "new_gate_train_dev_events.csv",
        ROOT / "new_final_holdout_events.csv",
    )
    dev.to_csv(dev_path, index=False)
    holdout.to_csv(final_path, index=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    freeze = {
        "created_before_any_future_gate_training_or_evaluation": True,
        "new_gate_train_dev_candidate_periods": len(dev),
        "new_final_holdout_candidate_periods": len(holdout),
        "development_manifest_sha256": digest(dev_path),
        "final_holdout_manifest_sha256": digest(final_path),
        "prior_stage_overlap": False,
        "object_membership_status": "not yet mined; date windows frozen, and radar availability/object validity must be reported without outcome-dependent replacement",
        "minimum_scientific_requirement": "retain substantially more than three independent positive events after availability/mining gates; otherwise acquire additional predeclared periods before gate development",
        "future_final_holdout_policy": "no learned or rule-based gate prediction until object membership, hard negatives, tensors, and hashes are frozen",
    }
    (ROOT / "split_freeze.json").write_text(json.dumps(freeze, indent=2))
    return freeze


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
