"""Automated split-overlap audit for frozen Stage 8 qualification manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _walk_events(value: Any, source: Path):
    rows = []
    if isinstance(value, dict):
        if {"id", "start_utc", "end_utc"}.issubset(value):
            rows.append({"event_id": str(value["id"]), "start": pd.Timestamp(value["start_utc"]), "end": pd.Timestamp(value["end_utc"]), "source": source.as_posix()})
        for child in value.values():
            rows.extend(_walk_events(child, source))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_walk_events(child, source))
    return rows


def run(root: Path = Path("artifacts/stage_8/qualification")) -> dict:
    dev = pd.read_csv(root / "stage8_development_object_manifest.csv")
    final = pd.read_csv(root / "stage8_final_object_manifest.csv")
    candidates = pd.read_csv(root / "stage8_candidate_qualification.csv")
    config = yaml.safe_load(Path("configs/data/stage_8_qualification.yaml").read_text(encoding="utf-8"))
    interval = {event["id"]: (pd.Timestamp(event["start_utc"]), pd.Timestamp(event["end_utc"])) for key in ("development_events", "final_events") for event in config[key]}
    prior = []
    for path in sorted(Path("configs/data").glob("*.yaml")):
        if path.name == "stage_8_qualification.yaml":
            continue
        prior.extend(_walk_events(yaml.safe_load(path.read_text(encoding="utf-8")), path))
    conflicts = []
    for row in candidates.itertuples(index=False):
        start, end = interval[row.event_id]
        for old in prior:
            if max(start, old["start"]) <= min(end, old["end"]):
                conflicts.append({"stage8_event_id": row.event_id, "stage8_split": row.split, "prior_event_id": old["event_id"], "prior_source": old["source"], "overlap_start": max(start, old["start"]).isoformat(), "overlap_end": min(end, old["end"]).isoformat()})
    conflict_table = pd.DataFrame(conflicts, columns=["stage8_event_id", "stage8_split", "prior_event_id", "prior_source", "overlap_start", "overlap_end"])
    conflict_table.to_csv(root / "stage8_prior_stage_overlap_audit.csv", index=False)
    checks = {
        "event_id_overlap": sorted(set(dev.event_id) & set(final.event_id)),
        "weather_system_group_overlap": sorted(set(dev.independent_system_group) & set(final.independent_system_group)),
        "issue_time_overlap": sorted(set(dev.issue_time_utc) & set(final.issue_time_utc)),
        "object_id_overlap": sorted(set(dev.object_id) & set(final.object_id)),
        "prior_stage_window_overlap_rows": len(conflict_table),
        "prior_configs_audited": sorted(set(row["source"] for row in prior)),
    }
    checks["all_disjoint"] = not any(checks[key] for key in ("event_id_overlap", "weather_system_group_overlap", "issue_time_overlap", "object_id_overlap")) and len(conflict_table) == 0
    (root / "stage8_disjointness_audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not checks["all_disjoint"]:
        raise RuntimeError("Stage 8 split/prior-stage overlap detected")
    return checks


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
