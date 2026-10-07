"""Post-score descriptive expected-rate diagnostics; never used for selection."""

from pathlib import Path

import numpy as np
import pandas as pd


def run() -> None:
    table = pd.read_csv("artifacts/stage_6/stage6_cache_manifest.csv")
    table = table[table.row_role == "heavy_rain_generalization"]
    rows = []
    for record in table.itertuples(index=False):
        source = np.load(record.tensor_path)
        cache = np.load(record.stage6_cache_path)
        observed = np.where(source["target_valid_mask"] > 0, source["target_rate_mm_hr"], np.nan)
        a_plus = cache["aplus_expected_rate_mm_hr"]
        hourly = cache["hrrr_apcp_hourly"]
        hrrr = np.stack([hourly[0]] * 10 + [hourly[1]] * 10)
        forecasts = {"A_plus": a_plus, "raw_HRRR": hrrr, "hybrid": 0.5 * a_plus + 0.5 * hrrr}
        for lead in (30, 60, 90, 120):
            index = lead // 6 - 1
            valid = np.isfinite(observed[index])
            for model, forecast in forecasts.items():
                rows.append(
                    {
                        "event_id": record.event_id,
                        "object_id": record.object_id,
                        "model": model,
                        "lead_minutes": lead,
                        "observed_mean_rate_mm_hr": float(np.mean(observed[index][valid])),
                        "forecast_mean_rate_mm_hr": float(np.mean(forecast[index][valid])),
                        "rate_mae_mm_hr": float(
                            np.mean(np.abs(forecast[index][valid] - observed[index][valid]))
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    output = Path("artifacts/stage_6/evaluation")
    result.to_csv(output / "heavy_rain_expected_rate_by_row.csv", index=False)
    result.groupby(["model", "lead_minutes"], as_index=False).mean(numeric_only=True).to_csv(
        output / "heavy_rain_expected_rate_summary.csv", index=False
    )


if __name__ == "__main__":
    run()
