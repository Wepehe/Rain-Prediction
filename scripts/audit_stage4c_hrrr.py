"""Audit the complete Stage 4C HRRR thermodynamic source gate."""

import collections
import json
from pathlib import Path

import eccodes
import pandas as pd

root = Path("artifacts/stage_4c/materialization")
materialized = pd.read_csv(root / "stage4_hrrr_materialized_sources.csv")
required_records = pd.read_csv(root / "stage4_hrrr_required_sources.csv")
required_fields = {"2t", "2d", "cape", "cin", "pwat"}
counts: collections.Counter[str] = collections.Counter()
failures = []
for row in materialized.itertuples(index=False):
    found = set()
    with Path(row.local_path).open("rb") as handle:
        while True:
            message = eccodes.codes_grib_new_from_file(handle)
            if message is None:
                break
            found.add(str(eccodes.codes_get(message, "shortName")))
            eccodes.codes_release(message)
    counts.update(found & required_fields)
    if not required_fields <= found:
        failures.append({"product_key": row.product_key, "missing": sorted(required_fields - found)})

forecast_issue = pd.to_datetime(required_records.forecast_issue_time_utc, utc=True)
model_issue = pd.to_datetime(required_records.model_issue_time_utc, utc=True)
model_valid = pd.to_datetime(required_records.model_valid_time_utc, utc=True)
report = {
    "products": len(materialized),
    "records": len(required_records),
    "variables": sorted(required_records.variable.unique()),
    "all_product_files_exist": bool(all(Path(path).exists() for path in materialized.local_path)),
    "missing_field_products": len(failures),
    "decoded_field_product_counts": dict(counts),
    "all_cycles_available_by_issue": bool((model_issue <= forecast_issue).all()),
    "future_valid_records_from_available_cycles": int((model_valid > forecast_issue).sum()),
    "negative_forecast_hours": int((required_records.forecast_hour < 0).sum()),
    "failures": failures,
    "source_gate_passed": bool(len(materialized) == 420 and not failures),
}
(root / "stage4c_thermodynamic_source_gate.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
