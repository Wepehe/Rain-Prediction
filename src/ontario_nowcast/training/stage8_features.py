"""Materialize frozen Stage 8 development features before gate fitting."""

from __future__ import annotations

import hashlib, json
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import torch
import yaml

from ..data.manifest import sha256_file
from ..events.tracking import build_independent_initiation_catalog
from ..models.baselines import pysteps_deterministic_extrapolation
from ..models.radar_convlstm import build_radar_model
from ..preprocessing.grid import transform_coordinates
from ..sample_evaluation import _downsample_max
from .stage4c_tensors import _apply, _weights

ROOT = Path("artifacts/stage_8")
PREDECL_SHA = "18e2d0273f5e3a04028339e5cd534d9b2e86a3a8129d32234aaa252e7e94ec5d"
FEATURES = ["lead_fraction","current_radar_wet_fraction","a_plus_wet_fraction","a_plus_mean_entropy","pysteps_wet_fraction","hrrr_wet_fraction","a_plus_hrrr_disagreement_fraction","hrrr_radar_forecast_support_fraction"]

def _apcp(path: Path, geometry=False):
    with path.open("rb") as f: m=eccodes.codes_grib_new_from_file(f)
    try:
        values=np.asarray(eccodes.codes_get_values(m),np.float32); geom=None
        if geometry:
            lat=np.asarray(eccodes.codes_get_array(m,"latitudes")); lon=(np.asarray(eccodes.codes_get_array(m,"longitudes"))+180)%360-180
            geom=transform_coordinates(lon,lat,source_crs="EPSG:4326",destination_crs="EPSG:3978")
        return values,geom
    finally: eccodes.codes_release(m)

def run():
    spec_path=ROOT/"gate_predeclaration/stage8_gate_predeclaration.json"
    if sha256_file(spec_path)!=PREDECL_SHA: raise RuntimeError("predeclaration hash mismatch")
    spec=json.loads(spec_path.read_text()); assert not spec["gate_trained"]
    pos=pd.read_csv(ROOT/"qualification/stage8_development_object_manifest.csv")
    neg=pd.read_csv(ROOT/"qualification/stage8_hard_negative_manifest.csv"); neg=neg[neg.split.eq("development")]
    members=pd.read_csv(ROOT/"gate_predeclaration/stage8_development_row_system_membership.csv")
    hrecords=pd.read_csv(ROOT/"qualification/stage8_hrrr_causal_records.csv")
    hsources=pd.read_csv(ROOT/"qualification/stage8_hrrr_source_manifest.csv").set_index("product_key")
    first=Path(hsources.iloc[0].local_path); _,(sx,sy)=_apcp(first,True)
    cfg=yaml.safe_load(Path("configs/experiments/stage_4b_a_plus_radar_capacity.yaml").read_text())
    meta=json.loads(Path("artifacts/stage_4b/a_plus_radar/training_metadata.json").read_text()); norm=meta["normalization"]["radar"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=build_radar_model({"model":cfg["model"],"data":cfg["data"]}).to(device)
    model.load_state_dict(torch.load(spec["immutable_inputs"]["a_plus"]["checkpoint"],map_location=device)); model.eval()
    out=ROOT/"development_features"; cache_dir=out/"row_cache"; cache_dir.mkdir(parents=True,exist_ok=True)
    event_cache={}; catalog_cache={}; geometry_weights={}; hrrr_values={}; manifest=[]; cohorts=[]; audit=[]
    all_mined=pd.read_csv(ROOT/"qualification/stage8_all_mined_objects.csv")
    def event(eid,source=None):
        if eid not in event_cache:
            p=Path(source) if source else next((Path(f"data/stage8_qualification/{eid}/processed/events/{eid}.npz"),Path(f"data/s8/e01/processed/events/{eid}.npz"),Path(f"data/s8/e02/processed/events/{eid}.npz")))
            if not p.exists(): p=next(x for x in [Path(f"data/stage8_qualification/{eid}/processed/events/{eid}.npz"),Path(f"data/s8/e01/processed/events/{eid}.npz"),Path(f"data/s8/e02/processed/events/{eid}.npz")] if x.exists())
            z=np.load(p); rates=_downsample_max(z["rate_mm_hr"],2); event_cache[eid]=(p,rates,pd.to_datetime(z["times"],utc=True),z["latitude"][:rates.shape[1]*2:2],z["longitude"][:rates.shape[2]*2:2])
        return event_cache[eid]
    rows=[]
    for r in pos.itertuples(index=False): rows.append((r.object_id,"positive_initiation",r.event_id,r.independent_system_group,r.issue_time_utc,r.tile_y_min,r.tile_y_max,r.tile_x_min,r.tile_x_max,r.radar_source_path))
    for r in neg.itertuples(index=False): rows.append((r.hard_negative_id,"hard_negative",r.event_id,r.independent_system_group,r.issue_time_utc,r.tile_y_min,r.tile_y_max,r.tile_x_min,r.tile_x_max,None))
    for n,(rid,role,eid,group,issue,y0,y1,x0,x1,source) in enumerate(rows,1):
        p,rates,times,lat,lon=event(eid,source); issue=pd.Timestamp(issue); a=times.get_indexer([issue])[0]
        hist=rates[a-9:a+1,int(y0):int(y1)+1,int(x0):int(x1)+1]; target=rates[a+1:a+21,int(y0):int(y1)+1,int(x0):int(x1)+1]
        valid=np.isfinite(target); rin=np.stack([((np.log1p(np.nan_to_num(hist))-norm["mean"])/norm["std"]),np.isfinite(hist).astype(np.float32)],1).astype(np.float32)
        with torch.no_grad(): logits,_=model(torch.from_numpy(rin[None]).to(device)); apl=torch.sigmoid(logits)[0,:,0].cpu().numpy().astype(np.float32)
        py=pysteps_deterministic_extrapolation(hist[-3:],20)
        key=(eid,int(y0),int(y1),int(x0),int(x1))
        if key not in geometry_weights:
            glon,glat=np.meshgrid(lon[int(x0):int(x1)+1],lat[int(y0):int(y1)+1]); tx,ty=transform_coordinates(glon,glat,source_crs="EPSG:4326",destination_crs="EPSG:3978"); geometry_weights[key]=_weights(sx,sy,tx,ty)
        vertices,weights=geometry_weights[key]; rec=hrecords[hrecords.issue_time_utc.eq(issue.isoformat())].sort_values("forecast_hour")
        hourly=[]
        for rr in rec.itertuples(index=False):
            if rr.product_identity not in hrrr_values: hrrr_values[rr.product_identity]=_apcp(Path(hsources.loc[rr.product_identity].local_path))[0]
            hourly.append(_apply(hrrr_values[rr.product_identity],vertices,weights,(128,128)))
        hr=np.stack([hourly[0]]*10+[hourly[1]]*10); hocc=hr>=.1; aw=apl>=.35; pw=py>=.1
        final_valid=np.isfinite(hist[-1]); feats=[]
        for j in range(20):
            pclip=np.clip(apl[j],1e-6,1-1e-6); hw=hocc[j]
            feats.append([(j+1)/20,float(np.mean((hist[-1]>=.1)[final_valid])),float(aw[j].mean()),float(np.mean(-pclip*np.log(pclip)-(1-pclip)*np.log(1-pclip))),float(pw[j].mean()),float(hw.mean()),float(np.mean(aw[j]^hw)),float(np.mean((aw[j]|pw[j])[hw])) if hw.any() else 0.])
        feats=np.asarray(feats,np.float32); observed=((target>=.1)&valid).astype(np.uint8)
        cache=cache_dir/f"{hashlib.sha256(rid.encode()).hexdigest()[:20]}.npz"
        np.savez_compressed(cache,features=feats,a_plus_probability=apl.astype(np.float16),pysteps_rate=py.astype(np.float16),hrrr_occurrence=hocc.astype(np.uint8),target_occurrence=observed,target_valid=valid.astype(np.uint8),target_rate=target.astype(np.float16))
        manifest.append({"row_id":rid,"role":role,"event_id":eid,"independent_system_group":group,"issue_time_utc":issue.isoformat(),"cache_path":cache.as_posix(),"cache_sha256":sha256_file(cache)})
        if role=="positive_initiation":
            eventual=np.any(observed.astype(bool),axis=0)&(hist[-1]<.1); vc=float(min(np.isfinite(hist).mean(),valid.mean())); dry=float(np.mean(hist[-1][eventual]<.1)) if eventual.any() else 0.; hwf=float(np.mean((hist>=.1)&np.isfinite(hist))); pycov=float(np.mean(np.any(py[:,eventual]>=.1,axis=0))) if eventual.any() else 1.; yy,xx=np.where(eventual); v2boundary=bool(eventual.any() and yy.min()>=9 and xx.min()>=9 and yy.max()<119 and xx.max()<119)
            if eid not in catalog_cache:
                catalog_cache[eid]=build_independent_initiation_catalog(rates,times,lat,lon,interval_minutes=6,resolution_km=2,history_minutes=60,horizon_minutes=120,rain_threshold=.1,strong_threshold=5,minimum_pixels=12)
            cat=catalog_cache[eid]
            track=rid.split("__")[-1]; obj=cat[cat.event_id.eq(track)].iloc[0]; limitedboundary=bool(obj.y_min-int(y0)>=9 and obj.x_min-int(x0)>=9 and int(y1)-obj.y_max>=9 and int(x1)-obj.x_max>=9)
            adv=False; limited=vc>=.95 and dry>=.90 and hwf<=.05 and pycov<=.40 and not adv and limitedboundary; v2=vc>=.95 and dry>=.95 and hwf<=.02 and pycov<=.20 and not adv and v2boundary
            cohorts.append({"row_id":rid,"event_id":eid,"independent_system_group":group,"valid_coverage_fraction":vc,"final_dry_fraction_at_eventual_pixels":dry,"history_wet_area_fraction":hwf,"pysteps_coverage_at_eventual_pixels":pycov,"boundary_clear_18km":limitedboundary,"radar_limited_initiation":limited,"radar_poor_initiation_v2":v2})
        audit.append({"row_id":rid,"issue_time_utc":issue.isoformat(),"source_latest_utc":issue.isoformat(),"hrrr_availability_max_utc":rec.simulated_availability_time_utc.max(),"all_sources_causal":bool(pd.Timestamp(rec.simulated_availability_time_utc.max())<=issue),"feature_count":8,"lead_count":20,"source_radar_sha256":sha256_file(p)})
        if n%10==0: print(f"[stage8-features] {n}/{len(rows)}",flush=True)
    m=pd.DataFrame(manifest); c=pd.DataFrame(cohorts); a=pd.DataFrame(audit)
    m.to_csv(out/"development_feature_manifest.csv",index=False); c.to_csv(out/"stage8_development_radar_cohorts.csv",index=False); a.to_csv(out/"feature_causality_audit.csv",index=False)
    freeze={"predeclaration_sha256":PREDECL_SHA,"rows":len(m),"positive_rows":int((m.role=="positive_initiation").sum()),"hard_negative_rows":int((m.role=="hard_negative").sum()),"features":FEATURES,"all_causal":bool(a.all_sources_causal.all()),"manifest_sha256":sha256_file(out/"development_feature_manifest.csv"),"cohort_sha256":sha256_file(out/"stage8_development_radar_cohorts.csv"),"audit_sha256":sha256_file(out/"feature_causality_audit.csv"),"radar_limited_rows":int(c.radar_limited_initiation.sum()),"v2_rows":int(c.radar_poor_initiation_v2.sum()),"gate_predictions_generated":False,"final_features_generated":False}
    (out/"development_feature_freeze.json").write_text(json.dumps(freeze,indent=2)); print(json.dumps(freeze,indent=2))

if __name__=="__main__": run()
