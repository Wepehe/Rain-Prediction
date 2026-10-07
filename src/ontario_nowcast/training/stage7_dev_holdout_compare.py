"""Compare descriptive HRRR support on Stage 5 DEV and consumed 2023 rows."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.onset import first_crossing_minutes


def run() -> None:
    rows = []
    stage5 = pd.read_csv("artifacts/stage_5/cache/stage5_cache_manifest.csv")
    stage5 = stage5[
        stage5.split.isin({"dev", "stage4_dev_repair"})
        & stage5.event_class.eq("positive_initiation")
    ]
    for row in stage5.itertuples(index=False):
        source = np.load(row.radar_goes_tensor_path)
        cache = np.load(row.stage5_cache_path)
        observed = np.where(source["target_valid_mask"] > 0, source["target_rate_mm_hr"], np.nan)
        footprint = np.any(observed >= 0.1, axis=0)
        onset = first_crossing_minutes(observed, 0.1, 6)
        median = float(np.nanmedian(onset[footprint]))
        hourly = cache["hrrr_apcp_hourly"]
        hmask = hourly[0 if median <= 60 else 1] >= 0.1
        rows.append(
            {
                "dataset": "stage5_dev",
                "event_id": row.event_id,
                "rows": 1,
                "hrrr_direct_overlap": float(np.any(hmask & footprint)),
                "hrrr_footprint_coverage": float(np.mean(hmask[footprint])),
                "onset_minute": median,
            }
        )
    holdout = pd.read_csv("artifacts/stage_7/row_level_error_table.csv")
    for row in holdout.itertuples(index=False):
        rows.append(
            {
                "dataset": "stage6_2023",
                "event_id": row.event_id,
                "rows": 1,
                "hrrr_direct_overlap": float(row.hrrr_correct_interval_at_location),
                "hrrr_footprint_coverage": row.hrrr_occurrence_fraction,
                "onset_minute": row.observed_onset_minute,
            }
        )
    table = pd.DataFrame(rows)
    output = Path("artifacts/stage_7")
    table.groupby("dataset", as_index=False).mean(numeric_only=True).to_csv(
        output / "dev_vs_2023_hrrr_support.csv", index=False
    )
    table.groupby(["dataset", "event_id"], as_index=False).mean(numeric_only=True).to_csv(
        output / "dev_vs_2023_hrrr_support_by_event.csv", index=False
    )


if __name__ == "__main__":
    run()
