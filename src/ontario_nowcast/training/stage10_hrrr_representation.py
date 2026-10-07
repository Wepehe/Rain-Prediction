"""Stage 10 causal HRRR representation and displacement diagnosis."""
from __future__ import annotations
import hashlib,json,urllib.request
from pathlib import Path
import eccodes,numpy as np,pandas as pd
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve
from scipy.stats import rankdata,spearmanr
from ..data.manifest import sha256_file
from ..evaluation.metrics import categorical_metrics,fractions_skill_score_km
from ..preprocessing.grid import transform_coordinates
from ..sample_evaluation import _downsample_max
from .stage4c_tensors import _apply,_weights

ROOT=Path("artifacts/stage_10"); CACHE=ROOT/"f01_apcp"; OUT=ROOT/"analysis"; MAX_SHIFT=18

def idx(url):
 return urllib.request.urlopen(url,timeout=60).read().decode().splitlines()
def message(lines,token):
 for i,line in enumerate(lines):
  if token in line:
   parts=line.split(":");start=int(parts[1]);end=int(lines[i+1].split(":")[1])-1 if i+1<len(lines) else None;return line,start,end
 return None
def get_range(url,start,end):
 req=urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}" if end else f"bytes={start}-"});return urllib.request.urlopen(req,timeout=120).read()
def grib(path,geometry=False):
 with path.open("rb") as f:m=eccodes.codes_grib_new_from_file(f)
 try:
  v=np.asarray(eccodes.codes_get_values(m),np.float32);meta={k:eccodes.codes_get(m,k) for k in ["shortName","name","units","stepType","stepRange","dataDate","dataTime","validityDate","validityTime"]};geom=None
  if geometry:
   lat=np.asarray(eccodes.codes_get_array(m,"latitudes"));lon=(np.asarray(eccodes.codes_get_array(m,"longitudes"))+180)%360-180;geom=transform_coordinates(lon,lat,source_crs="EPSG:4326",destination_crs="EPSG:3978")
  return v,meta,geom
 finally:eccodes.codes_release(m)
def shift(a,dy,dx):
 out=np.zeros_like(a); ys=slice(max(0,dy),min(a.shape[-2],a.shape[-2]+dy));xs=slice(max(0,dx),min(a.shape[-1],a.shape[-1]+dx));sy=slice(max(0,-dy),min(a.shape[-2],a.shape[-2]-dy));sx=slice(max(0,-dx),min(a.shape[-1],a.shape[-1]-dx));out[...,ys,xs]=a[...,sy,sx];return out
def best_shift(src,dst):
 s=src.astype(bool);d=dst.astype(bool)
 if not s.any() or not d.any():return 0,0,0.,False
 correlation=fftconvolve(d.astype(float),s[::-1,::-1].astype(float),mode="same")
 cy,cx=np.asarray(s.shape)//2;window=correlation[cy-MAX_SHIFT:cy+MAX_SHIFT+1,cx-MAX_SHIFT:cx+MAX_SHIFT+1];wy,wx=np.unravel_index(np.argmax(window),window.shape);dy,dx=int(wy-MAX_SHIFT),int(wx-MAX_SHIFT)
 q=shift(s,dy,dx);score=np.logical_and(q,d).sum()/max(1,np.logical_or(q,d).sum())
 return dy,dx,float(score),bool(score>=.05)
def auc(y,x):
 y=np.asarray(y,bool);n1=y.sum();n0=(~y).sum()
 if not n1 or not n0:return np.nan
 return (rankdata(x)[y].sum()-n1*(n1+1)/2)/(n1*n0)

def run():
 ROOT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 hr=pd.read_csv("artifacts/stage_8/qualification/stage8_hrrr_causal_records.csv"); cycles=hr[["hrrr_cycle_issue_time_utc"]].drop_duplicates();audit=[];sources=[]
 if (ROOT/"hrrr_representation_source_audit.csv").exists() and (ROOT/"f01_apcp_source_manifest.csv").exists():
  audit=pd.read_csv(ROOT/"hrrr_representation_source_audit.csv").to_dict("records");sources=pd.read_csv(ROOT/"f01_apcp_source_manifest.csv").to_dict("records")
 else:
  for n,r in enumerate(cycles.itertuples(index=False),1):
   t=pd.Timestamp(r.hrrr_cycle_issue_time_utc);base=f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{t:%Y%m%d}/conus/hrrr.t{t:%H}z.wrfsfcf01.grib2"; lines=idx(base+".idx")
   candidates={"continuous_APCP":"APCP:surface:0-1 hour acc fcst:","PRATE":"PRATE:surface:1 hour fcst:","REFC":"REFC:entire atmosphere:1 hour fcst:","REFD_1000m":"REFD:1000 m above ground:1 hour fcst:"}
   for name,tok in candidates.items():
    found=message(lines,tok);audit.append({"cycle":t.isoformat(),"candidate":name,"available":found is not None,"record":found[0] if found else "","forecast_hour":1,"valid_time":(t+pd.Timedelta(hours=1)).isoformat(),"simulated_availability":(t+pd.Timedelta(hours=1)).isoformat()})
   rec=message(lines,candidates["continuous_APCP"]); path=CACHE/f"hrrr.{t:%Y%m%d}.t{t:%H}z.f01.apcp.grib2"
   if not path.exists():path.write_bytes(get_range(base,rec[1],rec[2]))
   _,meta,_=grib(path);sources.append({"cycle":t.isoformat(),"path":path.as_posix(),"sha256":sha256_file(path),**meta})
   if n%20==0:print(f"[stage10-audit] {n}/{len(cycles)}",flush=True)
  pd.DataFrame(audit).to_csv(ROOT/"hrrr_representation_source_audit.csv",index=False);pd.DataFrame(sources).to_csv(ROOT/"f01_apcp_source_manifest.csv",index=False)

 pos=pd.read_csv("artifacts/stage_8/qualification/stage8_development_object_manifest.csv");neg=pd.read_csv("artifacts/stage_8/qualification/stage8_hard_negative_manifest.csv");neg=neg[neg.split.eq("development")]; fm=pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv").set_index("row_id"); hrs=hr.sort_values("forecast_hour"); src=pd.read_csv("artifacts/stage_8/qualification/stage8_hrrr_source_manifest.csv").set_index("product_key")
 rows=[]
 for r in pos.itertuples(index=False):rows.append((r.object_id,"positive",r.event_id,r.independent_system_group,pd.Timestamp(r.issue_time_utc),r.tile_y_min,r.tile_y_max,r.tile_x_min,r.tile_x_max,r.radar_source_path))
 for r in neg.itertuples(index=False):rows.append((r.hard_negative_id,"hard_negative",r.event_id,r.independent_system_group,pd.Timestamp(r.issue_time_utc),r.tile_y_min,r.tile_y_max,r.tile_x_min,r.tile_x_max,None))
 event_cache={};geo={};hval={}; diagnostics=[];metrics=[];failure=pd.read_csv("artifacts/stage_9/source_advantage/oracle_source_advantage_by_row_lead.csv");failure=failure[failure.collection.eq("stage8_nested_development")]
 for n,(rid,role,eid,group,issue,y0,y1,x0,x1,source) in enumerate(rows,1):
  if eid not in event_cache:
   opts=[Path(source)] if source else [] ;opts += [Path(f"data/stage8_qualification/{eid}/processed/events/{eid}.npz"),Path(f"data/s8/e01/processed/events/{eid}.npz"),Path(f"data/s8/e02/processed/events/{eid}.npz")];p=next(x for x in opts if x.exists());z=np.load(p);event_cache[eid]=(_downsample_max(z["rate_mm_hr"],2),pd.to_datetime(z["times"],utc=True),z["latitude"],z["longitude"])
  rates,times,lat0,lon0=event_cache[eid]; lat=lat0[:rates.shape[1]*2:2] if len(lat0)>rates.shape[1] else lat0;lon=lon0[:rates.shape[2]*2:2] if len(lon0)>rates.shape[2] else lon0;a=times.get_indexer([issue])[0];hist=rates[a-9:a+1,int(y0):int(y1)+1,int(x0):int(x1)+1];target=rates[a+1:a+21,int(y0):int(y1)+1,int(x0):int(x1)+1]
  recs=hrs[hrs.issue_time_utc.eq(issue.isoformat())];cycle=pd.Timestamp(recs.iloc[0].hrrr_cycle_issue_time_utc); key=(eid,int(y0),int(y1),int(x0),int(x1))
  f01path=CACHE/f"hrrr.{cycle:%Y%m%d}.t{cycle:%H}z.f01.apcp.grib2"
  if key not in geo:
   _,_,geom=grib(f01path,True);sx,sy=geom;glon,glat=np.meshgrid(lon[int(x0):int(x1)+1],lat[int(y0):int(y1)+1]);tx,ty=transform_coordinates(glon,glat,source_crs="EPSG:4326",destination_crs="EPSG:3978");geo[key]=_weights(sx,sy,tx,ty)
  vertices,weights=geo[key]
  f01=_apply(grib(f01path)[0],vertices,weights,(128,128)); future=[]
  for rr in recs.itertuples(index=False):
   if rr.product_identity not in hval:hval[rr.product_identity]=grib(Path(src.loc[rr.product_identity].local_path))[0]
   future.append(_apply(hval[rr.product_identity],vertices,weights,(128,128)))
  future=np.stack([future[0]]*10+[future[1]]*10);past_times=times[(times>cycle)&(times<=cycle+pd.Timedelta(hours=1))];inds=times.get_indexer(past_times);past=np.nansum(rates[inds,int(y0):int(y1)+1,int(x0):int(x1)+1]*.1,axis=0)
  dy,dx,match,meaningful=best_shift(f01>=.1,past>=.1);corrected=shift(future,dy,dx); mag=2*np.hypot(dy,dx)
  diagnostics.append({"row_id":rid,"role":role,"event_id":group,"issue_time_utc":issue.isoformat(),"shift_dy_pixels":dy,"shift_dx_pixels":dx,"shift_magnitude_km":mag,"direction_degrees":float(np.degrees(np.arctan2(dy,dx))),"match_iou":match,"meaningful_match":meaningful,"raw_current_overlap":float(np.logical_and(f01>=.1,past>=.1).sum()/max(1,np.logical_or(f01>=.1,past>=.1).sum())),"raw_current_fss18":fractions_skill_score_km(past,f01,.1,18,2)})
  for label,field in [("raw",future),("causal_shift",corrected)]:
   for lead0,lead1 in [(0,30),(30,60),(60,90),(90,120)]:
    js=np.arange(lead0//6,lead1//6);obs=np.nanmax(target[js],axis=0);fc=np.mean(field[js],axis=0);cm=categorical_metrics(obs,fc,.1);valid=np.isfinite(obs);b=float(np.mean(((fc[valid]>=.1).astype(float)-(obs[valid]>=.1).astype(float))**2));row={"row_id":rid,"role":role,"event_id":group,"representation":label,"lead_group":f"{lead0}-{lead1}","brier":b,"continuous_spearman":spearmanr(fc[valid],(obs[valid]>=.1).astype(float)).statistic,"continuous_auc":auc(obs[valid]>=.1,fc[valid]),**{k:cm[k] for k in ["csi","pod","far","f1"]}}
    for rad in [6,18,36]:row[f"fss_{rad}km"]=fractions_skill_score_km(obs,fc,.1,rad,2)
    metrics.append(row)
  if n%20==0:print(f"[stage10-analysis] {n}/{len(rows)}",flush=True)
 pd.DataFrame(diagnostics).to_csv(OUT/"causal_displacement_by_row.csv",index=False);pd.DataFrame(metrics).to_csv(OUT/"representation_metrics_by_row_lead_group.csv",index=False)
 d=pd.DataFrame(diagnostics);m=pd.DataFrame(metrics);d.groupby(["role","event_id"],as_index=False).agg(rows=("row_id","size"),mean_shift_km=("shift_magnitude_km","mean"),median_shift_km=("shift_magnitude_km","median"),mean_match_iou=("match_iou","mean"),meaningful_fraction=("meaningful_match","mean")).to_csv(OUT/"displacement_by_event.csv",index=False)
 ev=m.groupby(["role","event_id","representation"],as_index=False).mean(numeric_only=True);ev.to_csv(OUT/"skill_by_event.csv",index=False)
 summary=ev.groupby(["role","representation"],as_index=False).mean(numeric_only=True);summary.to_csv(OUT/"skill_event_macro.csv",index=False)
 # Join fixed Stage-9 failure categories to each row/lead group descriptively.
 fcat=failure.groupby("row_id").pysteps_failure_category.agg(lambda x:x.value_counts().index[0]); gain=ev[ev.role.eq("positive")].pivot(index="event_id",columns="representation",values="brier");
 auditdf=pd.DataFrame(audit);availability=auditdf.groupby("candidate").available.agg(["sum","count"]).reset_index();availability.to_csv(ROOT/"representation_availability_summary.csv",index=False)
 manifest={"status":"DIAGNOSTIC COMPLETE","fusion_model_trained":False,"sealed_final_accessed":False,"fixed_shift":{"algorithm":"exhaustive occurrence-IoU translation","max_abs_pixels":MAX_SHIFT,"resolution_km":2,"estimated_from":"f01 0-1h APCP versus matching past 1h radar accumulation"},"rows":len(rows),"outputs":{p.name:sha256_file(p) for p in list(ROOT.glob("*.csv"))+list(OUT.glob("*.csv"))}};(ROOT/"stage10_manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__":run()
