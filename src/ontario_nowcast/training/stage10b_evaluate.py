"""Evaluate native 15-minute HRRR representations on consumed Stage 8 development."""
from __future__ import annotations
import json
from pathlib import Path
import eccodes,numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
from ..data.manifest import sha256_file
from ..evaluation.metrics import categorical_metrics,fractions_skill_score_km
from ..preprocessing.grid import transform_coordinates
from ..sample_evaluation import _downsample_max
from .stage4c_tensors import _apply,_weights
ROOT=Path("artifacts/stage_10b");OUT=ROOT/"evaluation";OFF=np.arange(15,121,15)
def read(path,geom=False):
 with Path(path).open("rb") as f:m=eccodes.codes_grib_new_from_file(f)
 try:
  v=np.asarray(eccodes.codes_get_values(m),np.float32);g=None
  if geom:
   lat=np.asarray(eccodes.codes_get_array(m,"latitudes"));lon=(np.asarray(eccodes.codes_get_array(m,"longitudes"))+180)%360-180;g=transform_coordinates(lon,lat,source_crs="EPSG:4326",destination_crs="EPSG:3978")
  return v,g
 finally:eccodes.codes_release(m)
def auc(y,x):
 y=np.asarray(y,bool);n1=y.sum();n0=(~y).sum()
 return np.nan if not n1 or not n0 else (rankdata(x)[y].sum()-n1*(n1+1)/2)/(n1*n0)
def interp_frames(frames,minutes):
 pos=minutes/6-1;lo=max(0,int(np.floor(pos)));hi=min(len(frames)-1,int(np.ceil(pos)));w=pos-lo;return frames[lo]*(1-w)+frames[hi]*w
def metric(obs,fc):
 q=np.isfinite(obs)&np.isfinite(fc)
 if not np.any(q):return {k:np.nan for k in ["csi","pod","far","f1","mae","rank_correlation","occurrence_auc","brier","forecast_wet_fraction","observed_wet_fraction"]}
 cm=categorical_metrics(obs[q],fc[q],.1);o=obs[q]>=.1;f=fc[q]
 out={k:cm[k] for k in ["csi","pod","far","f1"]}|{"mae":float(np.mean(np.abs(f-obs[q]))),"rank_correlation":spearmanr(f,o.astype(float)).statistic,"occurrence_auc":auc(o,f),"brier":float(np.mean(((f>=.1).astype(float)-o.astype(float))**2)),"forecast_wet_fraction":float(np.mean(f>=.1)),"observed_wet_fraction":float(o.mean())}
 return out
def run():
 OUT.mkdir(parents=True,exist_ok=True);sel=pd.read_csv(ROOT/"wrfsubhf_selected_message_manifest.csv");hr=pd.read_csv("artifacts/stage_8/qualification/stage8_hrrr_causal_records.csv");pos=pd.read_csv("artifacts/stage_8/qualification/stage8_development_object_manifest.csv");neg=pd.read_csv("artifacts/stage_8/qualification/stage8_hard_negative_manifest.csv");neg=neg[neg.split.eq("development")]
 rows=[]
 for r in pos.itertuples(index=False):rows.append({"row_id":r.object_id,"role":"positive","event":r.event_id,"group":r.independent_system_group,"issue":pd.Timestamp(r.issue_time_utc),"y0":r.tile_y_min,"y1":r.tile_y_max,"x0":r.tile_x_min,"x1":r.tile_x_max,"source":r.radar_source_path})
 for r in neg.itertuples(index=False):rows.append({"row_id":r.hard_negative_id,"role":"hard_negative","event":r.event_id,"group":r.independent_system_group,"issue":pd.Timestamp(r.issue_time_utc),"y0":r.tile_y_min,"y1":r.tile_y_max,"x0":r.tile_x_min,"x1":r.tile_x_max,"source":None})
 rows=pd.DataFrame(rows);issue_cycle=hr.drop_duplicates("issue_time_utc").set_index("issue_time_utc").hrrr_cycle_issue_time_utc;rows["cycle"]=[pd.Timestamp(issue_cycle.loc[x.isoformat()]).isoformat() for x in rows.issue]
 events={};geometry={};metrics=[];onsets=[];fssrows=[]
 for ci,(cycle,rr) in enumerate(rows.groupby("cycle"),1):
  files=sel[sel.cycle.eq(cycle)].set_index(["offset_minutes","variable"]);fields={}
  first=files.iloc[0].path;_,(sx,sy)=read(first,True)
  for r in rr.itertuples(index=False):
   if r.event not in events:
    opts=([Path(r.source)] if isinstance(r.source,str) else [])+[Path(f"data/stage8_qualification/{r.event}/processed/events/{r.event}.npz"),Path(f"data/s8/e01/processed/events/{r.event}.npz"),Path(f"data/s8/e02/processed/events/{r.event}.npz")];p=next(x for x in opts if x.exists());z=np.load(p);events[r.event]=(_downsample_max(z["rate_mm_hr"],2),pd.to_datetime(z["times"],utc=True),z["latitude"],z["longitude"])
   rates,times,lat0,lon0=events[r.event];lat=lat0[:rates.shape[1]*2:2] if len(lat0)>rates.shape[1] else lat0;lon=lon0[:rates.shape[2]*2:2] if len(lon0)>rates.shape[2] else lon0;a=times.get_indexer([r.issue])[0];hist=rates[a-9:a+1,int(r.y0):int(r.y1)+1,int(r.x0):int(r.x1)+1];target=rates[a+1:a+21,int(r.y0):int(r.y1)+1,int(r.x0):int(r.x1)+1]
   glon,glat=np.meshgrid(lon[int(r.x0):int(r.x1)+1],lat[int(r.y0):int(r.y1)+1]);tx,ty=transform_coordinates(glon,glat,source_crs="EPSG:4326",destination_crs="EPSG:3978");vertices,weights=_weights(sx,sy,tx,ty)
   forecast={k:[] for k in ["subhourly_APCP","subhourly_PRATE","subhourly_REFC"]}
   for minute in OFF:
    for var,name in [("APCP","subhourly_APCP"),("PRATE","subhourly_PRATE"),("REFC","subhourly_REFC")]:
     key=(cycle,int(minute),var)
     if key not in fields:fields[key]=read(files.loc[(minute,var)].path)[0]
     tile=_apply(fields[key],vertices,weights,(128,128))
     if var=="APCP":tile=tile*4
     elif var=="PRATE":tile=tile*3600
     else:tile=np.power(np.power(10,np.maximum(tile,-20)/10)/200,1/1.6)
     forecast[name].append(tile)
   forecast={k:np.stack(v) for k,v in forecast.items()};truth_rate=np.stack([interp_frames(target,m) for m in OFF]);cum=np.concatenate([np.zeros_like(target[:1]),np.cumsum(np.nan_to_num(target)*.1,axis=0)]);truth_ap=[]
   for end in OFF:
    ce=interp_frames(cum,end+6);cs=np.zeros_like(ce) if end==15 else interp_frames(cum,end-15+6);truth_ap.append((ce-cs)*4)
   truth_ap=np.stack(truth_ap)
   for name,fc in forecast.items():
    obs=truth_ap if name=="subhourly_APCP" else truth_rate
    for j,minute in enumerate(OFF):metrics.append({"row_id":r.row_id,"role":r.role,"event_id":r.group,"representation":name,"offset_minutes":minute,"lead_group":f"{(minute-1)//30*30}-{((minute-1)//30+1)*30}",**metric(obs[j],fc[j])})
    for gi,(start,end) in enumerate([(0,30),(30,60),(60,90),(90,120)]):
     js=slice(gi*2,gi*2+2);oo=np.nanmax(obs[js],axis=0);ff=np.nanmax(fc[js],axis=0);fssrows.append({"row_id":r.row_id,"role":r.role,"event_id":r.group,"representation":name,"lead_group":f"{start}-{end}",**{f"fss_{rad}km":fractions_skill_score_km(oo,ff,.1,rad,2) for rad in [6,18,36]}})
    dry=np.isfinite(hist[-1])&(hist[-1]<.1);eventual=np.any(truth_rate>=.1,axis=0)&dry;o=np.argmax(truth_rate>=.1,axis=0);pidx=np.argmax(fc>=.1,axis=0);oh=np.any(truth_rate>=.1,axis=0);ph=np.any(fc>=.1,axis=0);both=eventual&ph
    onset_err=(OFF[pidx[both]]-OFF[o[both]]) if both.any() else np.array([]);onsets.append({"row_id":r.row_id,"role":r.role,"event_id":r.group,"representation":name,"onset_mae":float(np.mean(np.abs(onset_err))) if len(onset_err) else np.nan,"onset_bias":float(np.median(onset_err)) if len(onset_err) else np.nan,"detected_fraction":float(np.mean(ph[eventual])) if eventual.any() else np.nan,"false_initiation_fraction":float(np.mean(ph[dry&~oh])) if np.any(dry&~oh) else np.nan,**{f"detection_by_{w}":float(np.mean(ph[eventual]&((OFF[pidx[eventual]]<=w)))) if eventual.any() else np.nan for w in [30,60,90,120]}})
  if ci%20==0:print(f"[stage10b-eval] cycles {ci}/{rows.cycle.nunique()}",flush=True)
 met=pd.DataFrame(metrics);ons=pd.DataFrame(onsets);fss=pd.DataFrame(fssrows);met.to_csv(OUT/"native_metrics_by_row_offset.csv",index=False);ons.to_csv(OUT/"onset_by_row.csv",index=False);fss.to_csv(OUT/"fss_by_row_lead_group.csv",index=False)
 ev=met.groupby(["role","event_id","representation","lead_group"],as_index=False).mean(numeric_only=True);ev.to_csv(OUT/"metrics_by_event_lead_group.csv",index=False);ev.groupby(["role","representation","lead_group"],as_index=False).mean(numeric_only=True).to_csv(OUT/"metrics_event_macro.csv",index=False)
 fssev=fss.groupby(["role","event_id","representation","lead_group"],as_index=False).mean(numeric_only=True);fssev.to_csv(OUT/"fss_by_event_lead_group.csv",index=False);fssev.groupby(["role","representation","lead_group"],as_index=False).mean(numeric_only=True).to_csv(OUT/"fss_event_macro.csv",index=False)
 onev=ons.groupby(["role","event_id","representation"],as_index=False).mean(numeric_only=True);onev.to_csv(OUT/"onset_by_event.csv",index=False);onev.groupby(["role","representation"],as_index=False).mean(numeric_only=True).to_csv(OUT/"onset_event_macro.csv",index=False)
 manifest={"status":"DIAGNOSTIC COMPLETE","fusion_model_trained":False,"sealed_final_accessed":False,"native_offsets_minutes":OFF.tolist(),"reflectivity_conversion":"Marshall-Palmer Z=200 R^1.6, fixed before scoring","apcp_semantics":"native 15-minute interval accumulation converted to equivalent mean mm/h and compared with interval-accumulated MRMS","outputs":{p.name:sha256_file(p) for p in OUT.glob("*.csv")}};(ROOT/"stage10b_evaluation_manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=="__main__":run()
