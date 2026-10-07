"""Run one frozen Residual V1 operational nowcast from an NPZ radar input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ontario_nowcast.operational.inference import forecast, load_operational_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/operational/residual_v1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--initialization-time", help="Override final timestamp (ISO-8601); preserves 6-min cadence")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    with np.load(args.input, allow_pickle=False) as source:
        required = {"history_rate", "timestamps", "validity_mask"}
        missing = required - set(source.files)
        if missing:
            parser.error(f"input is missing arrays: {sorted(missing)}")
        history = source["history_rate"]
        timestamps = source["timestamps"].astype(str).tolist()
        validity = source["validity_mask"]
        metadata = json.loads(str(source["metadata_json"])) if "metadata_json" in source else {}
    if args.initialization_time:
        from datetime import datetime, timedelta
        end = datetime.fromisoformat(args.initialization_time)
        timestamps = [(end - timedelta(minutes=6 * i)).isoformat() for i in range(9, -1, -1)]
    model = load_operational_model(args.bundle, device=args.device)
    result = forecast(model, history, timestamps, validity, metadata, diagnostics=args.diagnostics)
    args.output.mkdir(parents=True, exist_ok=True)
    product = result.save_npz(args.output / "forecast.npz")
    (args.output / "forecast_summary.json").write_text(
        json.dumps(result.summary(), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"forecast": str(product), **result.metadata["timing_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
