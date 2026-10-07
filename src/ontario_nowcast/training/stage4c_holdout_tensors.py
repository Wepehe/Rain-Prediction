"""Materialize exact radar/thermodynamic/target tensors for the frozen Stage C holdout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..preprocessing.grid import transform_coordinates
from .stage4_gate import _load_stage4_radar
from .stage4c_tensors import SHORT_NAMES, VARIABLES, _apply, _decode, _weights


def run() -> dict[str, object]:
    root=Path("artifacts/stage_4c/frozen_holdout"); manifest_path=root/"stage4c_final_holdout_manifest.csv"
    freeze=json.loads((root/"stage4c_final_holdout_freeze.json").read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=freeze["manifest_sha256"]: raise RuntimeError("frozen manifest changed")
    manifest=pd.read_csv(manifest_path); required=pd.read_csv("artifacts/stage_4c/holdout_sources/stage4_hrrr_required_sources.csv")
    sources=pd.read_csv("artifacts/stage_4c/holdout_sources/stage4_hrrr_materialized_sources.csv").set_index("product_key")
    interp=pd.read_csv("artifacts/stage_4c/holdout_sources/stage4c_holdout_interpolation.csv")
    first=Path(sources.iloc[0].local_path); _,geometry=_decode(first,include_geometry=True); assert geometry is not None
    sx,sy=geometry; weight_cache={}; field_cache={}; event_cache={}; rows=[]; failures=[]
    out=root/"tensors"
    for number,row in enumerate(manifest.itertuples(index=False),1):
      identity=f"{row.source_event_id}__{row.object_id}__{pd.Timestamp(row.representative_issue_time_utc):%Y%m%dT%H%M%SZ}"
      try:
        if row.source_event_id not in event_cache: event_cache[row.source_event_id]=_load_stage4_radar(row.source_event_id,Path("data"))
        rates,times,lat,lon=event_cache[row.source_event_id]; issue=pd.Timestamp(row.representative_issue_time_utc); lookup={t:i for i,t in enumerate(times)}
        history=[issue-pd.Timedelta(minutes=m) for m in (54,48,42,36,30,24,18,12,6,0)]; future=[issue+pd.Timedelta(minutes=m) for m in range(6,121,6)]
        if any(t not in lookup for t in history+future): raise RuntimeError("missing exact radar time")
        y0,y1,x0,x1=map(int,(row.tile_y_min,row.tile_y_max,row.tile_x_min,row.tile_x_max)); shape=(y1-y0+1,x1-x0+1)
        if shape!=(128,128): raise RuntimeError(f"invalid tile {shape}")
        radar=np.stack([rates[lookup[t],y0:y1+1,x0:x1+1] for t in history]); target=np.stack([rates[lookup[t],y0:y1+1,x0:x1+1] for t in future])
        glon,glat=np.meshgrid(lon[x0:x1+1],lat[y0:y1+1]); tx,ty=transform_coordinates(glon,glat,source_crs="EPSG:4326",destination_crs="EPSG:3978")
        key=(row.source_event_id,y0,y1,x0,x1)
        if key not in weight_cache: weight_cache[key]=_weights(sx,sy,tx,ty)
        vertices,w=weight_cache[key]; contexts=[]; provenance=[]
        selection=interp[(interp.event_id==row.source_event_id)&(interp.object_id==row.object_id)&(pd.to_datetime(interp.forecast_issue_time_utc,utc=True)==issue)]
        for context in selection.sort_values("context_offset_minutes").itertuples(index=False):
          fields=[]
          for valid in (context.lower_valid_time_utc,context.upper_valid_time_utc):
            req=required[(required.event_id==row.source_event_id)&(required.anchor_id==row.object_id)&(required.model_valid_time_utc==valid)]
            product=str(req.product_key.iloc[0]); path=Path(sources.loc[product].local_path)
            if product not in field_cache: field_cache[product]=_decode(path)[0]
            fields.append(np.stack([_apply(field_cache[product][SHORT_NAMES[v]],vertices,w,shape) for v in VARIABLES]))
          alpha=float(context.upper_weight); contexts.append((1-alpha)*fields[0]+alpha*fields[1]); provenance.append({"model_issue_time_utc":context.model_issue_time_utc,"requested_valid_time_utc":context.requested_valid_time_utc,"lower_valid_time_utc":context.lower_valid_time_utc,"upper_valid_time_utc":context.upper_valid_time_utc,"upper_weight":alpha})
        thermo=np.stack(contexts).astype(np.float32); destination=out/f"{identity}.npz"; destination.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination,radar_rate_mm_hr=radar.astype(np.float32),radar_valid_mask=np.isfinite(radar),thermodynamics=thermo,thermodynamics_valid_mask=np.isfinite(thermo),target_rate_mm_hr=target.astype(np.float32),target_valid_mask=np.isfinite(target),latitude=glat,longitude=glon,forecast_issue_time_utc=np.asarray(issue.isoformat()),object_id=np.asarray(row.object_id),event_id=np.asarray(row.source_event_id),tile_indices=np.asarray([y0,y1,x0,x1]),variable_names=np.asarray(VARIABLES),hrrr_provenance_json=np.asarray(json.dumps(provenance)),normalization_sha256=np.asarray(Path("artifacts/stage_4c/gate/thermodynamic_normalization_train_only.sha256").read_text().strip()))
        rows.append({"identity":identity,"event_id":row.source_event_id,"object_id":row.object_id,"row_role":row.row_role,"tensor_path":destination.as_posix(),"sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),"radar_valid_fraction":float(np.isfinite(radar).mean()),"thermodynamic_valid_fraction":float(np.isfinite(thermo).mean()),"target_valid_fraction":float(np.isfinite(target).mean())})
        print(f"[stage4c-holdout-tensors] {number}/{len(manifest)}",flush=True)
      except Exception as exc:  # noqa: BLE001 - preserve every row-level materialization failure
        failures.append({"identity":identity,"error":str(exc)})
    table=pd.DataFrame(rows); table.to_csv(root/"exact_tensor_manifest.csv",index=False); pd.DataFrame(failures).to_csv(root/"exact_tensor_failures.csv",index=False)
    result={"frozen_manifest_sha256":freeze["manifest_sha256"],"requested_rows":len(manifest),"materialized_rows":len(table),"failures":len(failures),"all_roles_materialized":len(table)==len(manifest),"later_hrrr_cycle_used":False,"future_radar_in_inputs":False,"c1_authorized":bool(len(table)==len(manifest) and not failures)}
    (root/"pretraining_integrity.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result

if __name__=="__main__": print(json.dumps(run(),indent=2))
