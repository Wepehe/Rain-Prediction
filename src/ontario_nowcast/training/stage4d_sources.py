"""Build and materialize the issue-safe Stage 4D TRAIN/DEV HRRR source gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .stage4_materialize import (
    HRRR_BUCKET,
    _download_hrrr_subset,
    _hrrr_message_ranges,
    _hrrr_required_product_records,
    _hrrr_spotcheck_rows,
)

NATIVE_VARIABLES = (
    "UGRD_10m",
    "VGRD_10m",
    "UGRD_850hPa",
    "VGRD_850hPa",
    "UGRD_500hPa",
    "VGRD_500hPa",
    "VVEL_700hPa",
)
ALLOWED_SPLITS = {"train", "dev", "stage4_dev_repair"}


def build_audit(output_dir: Path) -> pd.DataFrame:
    source = pd.read_csv("artifacts/stage_4c/materialization/stage4_hrrr_required_sources.csv")
    source = source[source.split.isin(ALLOWED_SPLITS)].copy()
    bases = source[
        [
            "event_id",
            "split",
            "anchor_id",
            "forecast_issue_time_utc",
            "model_issue_time_utc",
            "model_valid_time_utc",
        ]
    ].drop_duplicates()
    rows = [
        {**row, "modality": "nwp", "variable": variable}
        for row in bases.to_dict("records")
        for variable in NATIVE_VARIABLES
    ]
    audit = pd.DataFrame(rows).sort_values(
        ["split", "event_id", "anchor_id", "model_valid_time_utc", "variable"]
    )
    if not set(audit.split.unique()) <= ALLOWED_SPLITS or audit.anchor_id.nunique() != 72:
        raise RuntimeError("Stage 4D audit must contain only the 72 TRAIN/DEV anchors")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "stage4d_hrrr_audit.csv", index=False)
    return audit


def materialize(output_dir: Path, max_products: int | None = None) -> dict[str, Any]:
    audit = build_audit(output_dir)
    required = _hrrr_required_product_records(audit)
    products = (
        required[["product_key", "product", "model_issue_time_utc", "forecast_hour"]]
        .drop_duplicates()
        .sort_values(["model_issue_time_utc", "forecast_hour", "product"])
        .reset_index(drop=True)
    )
    all_keys = products.product_key.astype(str).tolist()
    if max_products is not None:
        products = products.iloc[:max_products]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    ranges: list[dict[str, Any]] = []
    spots: list[dict[str, Any]] = []

    def one(product: pd.Series) -> dict[str, Any]:
        key = str(product.product_key)
        try:
            subset = required[required.product_key.astype(str).eq(key)]
            selectors = {
                str(row.variable): str(row.message_selector) for row in subset.itertuples()
            }
            message_ranges = _hrrr_message_ranges(key, selectors)
            destination = (
                Path("data/raw/nwp/hrrr/stage4d") / key.replace("/conus/", "/")
            ).with_suffix(".stage4d_subset.grib2")
            record = _download_hrrr_subset(
                key, message_ranges, destination, Path("data/metadata/downloads.jsonl")
            )
            return {
                "row": {
                    "bucket": HRRR_BUCKET,
                    "product_key": key,
                    "local_path": record["local_path"],
                    "exists": destination.exists(),
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                    "cached": record["cached"],
                    "variables": json.dumps(sorted(selectors)),
                    "message_count": len(message_ranges),
                },
                "ranges": [{"product_key": key, **item} for item in message_ranges],
            }
        except Exception as exc:  # noqa: BLE001
            return {"failure": {"product_key": key, "error": str(exc)}}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(one, row) for _, row in products.iterrows()]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if "failure" in result:
                failures.append(result["failure"])
            else:
                rows.append(result["row"])
                ranges.extend(result["ranges"])
            if completed == 1 or completed % 10 == 0:
                print(f"[stage4d-sources] {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda item: item["product_key"])
    for row in rows[:4]:
        spots.extend(_hrrr_spotcheck_rows(Path(row["local_path"]), max_messages=8))
    pd.DataFrame(required).to_csv(output_dir / "stage4d_hrrr_required_sources.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "stage4d_hrrr_materialized_sources.csv", index=False)
    pd.DataFrame(ranges).to_csv(output_dir / "stage4d_hrrr_message_ranges.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "stage4d_hrrr_source_failures.csv", index=False)
    pd.DataFrame(spots).to_csv(output_dir / "stage4d_hrrr_decode_spotcheck.csv", index=False)
    complete = max_products is None and not failures and len(rows) == len(all_keys)
    summary = {
        "audit_sha256": hashlib.sha256(
            (output_dir / "stage4d_hrrr_audit.csv").read_bytes()
        ).hexdigest(),
        "anchors": int(audit.anchor_id.nunique()),
        "splits": sorted(audit.split.unique()),
        "required_records": len(required),
        "native_variables": list(NATIVE_VARIABLES),
        "unique_products": len(all_keys),
        "attempted_products": len(products),
        "materialized_products": len(rows),
        "failures": len(failures),
        "complete": complete,
        "protected_2023_holdout_used": False,
        "training_started": False,
    }
    (output_dir / "stage4d_source_gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_4d/source_gate"))
    parser.add_argument("--max-products", type=int)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output_dir, args.max_products), indent=2))


if __name__ == "__main__":
    main()
