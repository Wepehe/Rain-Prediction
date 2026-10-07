"""Freeze the fresh Stage C holdout using radar/object evidence only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.baselines import pysteps_deterministic_extrapolation
from .stage4_gate import _load_stage4_radar

RAIN = .1


def _cohorts(row: pd.Series) -> dict[str, object]:
    rates, times, _, _ = _load_stage4_radar(str(row.source_event_id), Path("data"))
    issue = pd.Timestamp(row.representative_issue_time_utc)
    lookup = {time: index for index, time in enumerate(times)}
    history_times = [issue-pd.Timedelta(minutes=m) for m in (54,48,42,36,30,24,18,12,6,0)]
    target_times = [issue+pd.Timedelta(minutes=m) for m in range(6,121,6)]
    wanted = history_times + target_times
    exact = all(time in lookup for time in wanted)
    y0,y1,x0,x1 = map(int,(row.tile_y_min,row.tile_y_max,row.tile_x_min,row.tile_x_max))
    if not exact:
        return {"radar_valid_fraction": 0., "radar_limited_initiation": False,
                "radar_poor_initiation_v2": False, "exact_radar_times_available": False}
    radar=np.stack([rates[lookup[t],y0:y1+1,x0:x1+1] for t in history_times])
    target=np.stack([rates[lookup[t],y0:y1+1,x0:x1+1] for t in target_times])
    valid=float(min(np.isfinite(radar).mean(),np.isfinite(target).mean()))
    eventual=np.any(target>=RAIN,axis=0)&(radar[-1]<RAIN)
    dry=float(np.mean(radar[-1][eventual]<RAIN)) if eventual.any() else 0.
    wet=float(np.mean((radar>=RAIN)&np.isfinite(radar)))
    py=pysteps_deterministic_extrapolation(radar[-3:],20)
    pycov=float(np.mean(np.any(py[:,eventual]>=RAIN,axis=0))) if eventual.any() else 1.
    boundary=bool(float(row.y_min)-y0>=9 and float(row.x_min)-x0>=9 and
                  y1-float(row.y_max)>=9 and x1-float(row.x_max)>=9)
    common=valid>=.95 and not bool(row.advective_entry_like) and boundary
    return {"radar_valid_fraction":valid,"final_dry_fraction":dry,
            "history_wet_fraction":wet,"pysteps_coverage":pycov,
            "boundary_clear_18km":boundary,"exact_radar_times_available":True,
            "radar_limited_initiation":bool(common and dry>=.90 and wet<=.05 and pycov<=.40),
            "radar_poor_initiation_v2":bool(common and dry>=.95 and wet<=.02 and pycov<=.20)}


def run() -> dict[str, object]:
    source=Path("artifacts/stage_4c/holdout_gate/event_object_validation.csv")
    objects=pd.read_csv(source)
    clean=objects[(objects.event_class=="positive_initiation") & objects.clean_pre_radar_initiation.astype(bool)].copy()
    rows=[]
    for i,(_,row) in enumerate(clean.iterrows(),1):
        rows.append({"row_role":"clean_initiation",**row.to_dict(),**_cohorts(row)})
        print(f"[stage4c-freeze] initiation {i}/{len(clean)}",flush=True)
    accepted=("stage4c_holdout_july19_2023_dry_candidate","stage4c_holdout_aug02_2023_dry_candidate")
    general=("stage4c_holdout_aug23_2023_heavy_rain",)
    # Four predeclared, evenly spaced exact anchors per retained event; centre tile.
    for role,event_ids in (("hard_negative",accepted),("heavy_rain_generalization",general)):
        for event_id in event_ids:
            rates,times,_,_=_load_stage4_radar(event_id,Path("data"))
            valid_times=times[10:-20]
            picks=np.linspace(0,len(valid_times)-1,4,dtype=int)
            for number in picks:
                issue=valid_times[number]; ny,nx=rates.shape[1:]
                y0=max(0,min(ny-128,ny//2-64)); x0=max(0,min(nx-128,nx//2-64))
                rows.append({"row_role":role,"source_event_id":event_id,
                    "object_id":f"{role}_{number:04d}","representative_issue_time_utc":issue.isoformat(),
                    "tile_y_min":y0,"tile_y_max":y0+127,"tile_x_min":x0,"tile_x_max":x0+127,
                    "event_class":role,"clean_pre_radar_initiation":False,
                    "advective_entry_like":False,"radar_valid_fraction":float(np.isfinite(rates[number+10-9:number+10+21,y0:y0+128,x0:x0+128]).mean()),
                    "exact_radar_times_available":True,"radar_limited_initiation":False,
                    "radar_poor_initiation_v2":False})
    table=pd.DataFrame(rows)
    out=Path("artifacts/stage_4c/frozen_holdout"); out.mkdir(parents=True,exist_ok=True)
    path=out/"stage4c_final_holdout_manifest.csv"; table.to_csv(path,index=False)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    init=table[table.row_role=="clean_initiation"]
    summary={"manifest_sha256":digest,"total_rows":len(table),"total_initiation_rows":len(init),
      "clean_initiation_rows":len(init),"unique_issue_times":int(init.representative_issue_time_utc.nunique()),
      "independent_positive_events":int(init.source_event_id.nunique()),
      "radar_limited_rows":int(init.radar_limited_initiation.sum()),
      "radar_limited_unique_issue_times":int(init[init.radar_limited_initiation].representative_issue_time_utc.nunique()),
      "radar_limited_independent_events":int(init[init.radar_limited_initiation].source_event_id.nunique()),
      "radar_poor_v2_rows":int(init.radar_poor_initiation_v2.sum()),
      "accepted_hard_negative_rows":int((table.row_role=="hard_negative").sum()),
      "accepted_hard_negative_events":2,"heavy_rain_generalization_rows":4,
      "environmental_matching":"incomplete: legacy full-field source retrieval produced no output for over 12 minutes; retained on passed radar sparse-dry gate per protocol",
      "c1_predictions_before_freeze":False}
    (out/"stage4c_final_holdout_freeze.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

if __name__ == "__main__": print(json.dumps(run(),indent=2))
