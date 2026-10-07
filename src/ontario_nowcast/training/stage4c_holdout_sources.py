"""Prepare and materialize issue-safe HRRR sources for the frozen Stage C holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .stage4_materialize import materialize_hrrr_nwp_sources
from .stage4c_tensors import VARIABLES


def run(output_dir: Path) -> dict[str, object]:
    manifest = Path("artifacts/stage_4c/frozen_holdout/stage4c_final_holdout_manifest.csv")
    expected = json.loads(
        Path("artifacts/stage_4c/frozen_holdout/stage4c_final_holdout_freeze.json").read_text()
    )["manifest_sha256"]
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("frozen Stage C holdout manifest hash changed")
    rows = []
    interpolation = []
    for record in pd.read_csv(manifest).itertuples(index=False):
        issue = pd.Timestamp(record.representative_issue_time_utc).tz_convert("UTC")
        # A one-hour latency is conservative and matches the established Stage 4C source gate.
        model_issue = issue.floor("h") - pd.Timedelta(hours=1)
        for offset in (0, 60, 120):
            requested = issue + pd.Timedelta(minutes=offset)
            lower, upper = requested.floor("h"), requested.ceil("h")
            weight = (requested-lower).total_seconds()/3600
            interpolation.append({"object_id":record.object_id,"event_id":record.source_event_id,
                "forecast_issue_time_utc":issue.isoformat(),"context_offset_minutes":offset,
                "requested_valid_time_utc":requested.isoformat(),"lower_valid_time_utc":lower.isoformat(),
                "upper_valid_time_utc":upper.isoformat(),"upper_weight":weight,
                "model_issue_time_utc":model_issue.isoformat()})
            for valid in sorted({lower,upper}):
                for variable in VARIABLES:
                    rows.append({"event_id":record.source_event_id,"split":"stage4c_final_holdout",
                        "anchor_id":record.object_id,"forecast_issue_time_utc":issue.isoformat(),
                        "modality":"nwp","variable":variable,
                        "model_issue_time_utc":model_issue.isoformat(),
                        "model_valid_time_utc":valid.isoformat()})
    output_dir.mkdir(parents=True,exist_ok=True)
    audit=output_dir/"stage4c_holdout_hrrr_audit.csv"
    pd.DataFrame(rows).drop_duplicates().to_csv(audit,index=False)
    pd.DataFrame(interpolation).to_csv(output_dir/"stage4c_holdout_interpolation.csv",index=False)
    result=materialize_hrrr_nwp_sources(audit,output_dir,data_root=Path("data"),
        active_only=False,variables=list(VARIABLES))
    result.update({"frozen_manifest_sha256":actual,"later_cycle_used":False,
        "temporal_interpolation":"linear between bracketing hourly valid fields from one already-issued cycle"})
    (output_dir/"stage4c_holdout_source_gate.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=Path("artifacts/stage_4c/holdout_sources"))
    args=parser.parse_args(); print(json.dumps(run(args.output_dir),indent=2))


if __name__ == "__main__": main()
