"""Build and freeze the Cycle-2 independent MRMS event corpus.

Phases are deliberately ordered: inventory -> screen -> materialize -> freeze.
No forecast model is called by this script.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd, requests, yaml
from scipy import ndimage

from ontario_nowcast.data.manifest import download_immutable, sha256_file
from ontario_nowcast.data.mrms import BUCKET, daily_objects
from ontario_nowcast.data.sample import _timestamp_from_key, decode_mrms_crop
from ontario_nowcast.data.s3 import object_url
from ontario_nowcast.events.tracking import build_independent_initiation_catalog
from ontario_nowcast.sample_evaluation import _downsample_max
from ontario_nowcast.training.stage4_gate import _tile_bounds, _tile_wgs84

ROOT=Path("artifacts/cycle2/data"); DATA=Path("data/cycle2"); BBOX=[-84.8,41.5,-75.5,46.5]
SEASONS={12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}
def _screen_decode(x):
 date,t,o,p=x
 try:
  a,_,_=decode_mrms_crop(p,BBOX);return {"date":date,"time":t.isoformat(),"wet_fraction":float(np.nanmean(a>=.1)),"heavy_fraction":float(np.nanmean(a>=5)),"max_rate":float(np.nanmax(a)),"finite_fraction":float(np.isfinite(a).mean()),"source_key":o.key,"sha256":sha256_file(p)}
 except Exception as e:return {"date":date,"time":t.isoformat(),"error":str(e)}
def _materialize_event(x):
 config_path,event_id,output_root=x
 from ontario_nowcast.data.sample import fetch_event
 processed=Path(output_root)/"processed"/"events"/f"{event_id}.npz"
 if processed.exists():return str(processed)
 return str(fetch_event(Path(config_path),event_id,Path(output_root),6))
def legacy_records():
 records=[]
 for p in [Path("configs/data/stage_8_qualification.yaml"),Path("configs/data/milestone_1_5.yaml")]:
  c=yaml.safe_load(p.read_text(encoding="utf-8"))
  for key in ["sample_events","events","hard_negative_events","development_events","final_events"]:
   for e in c.get(key,[]):
    if "start_utc" in e:
     records.append({"event_id":e.get("id","unknown"),"date":str(pd.Timestamp(e["start_utc"]).date()),"source_config":p.as_posix(),"source_section":key})
 for event_id,d in [
  ("stage4c_holdout_july12_2023_storms","2023-07-12"),("stage4c_holdout_july20_2023_supercells","2023-07-20"),
  ("stage4c_holdout_aug03_2023_severe_storms","2023-08-03"),("stage4c_holdout_july19_2023_dry_candidate","2023-07-19"),
  ("stage4c_holdout_aug02_2023_dry_candidate","2023-08-02"),("stage4c_holdout_aug23_2023_heavy_rain","2023-08-23")]:
  records.append({"event_id":event_id,"date":d,"source_config":"consumed_stage6_holdout","source_section":"consumed_2023"})
 return sorted({(x["event_id"],x["date"],x["source_config"],x["source_section"]):x for x in records}.values(),key=lambda x:(x["date"],x["event_id"]))
def inventory():
 ROOT.mkdir(parents=True,exist_ok=True);legacy_records_=legacy_records();legacy={pd.Timestamp(x["date"]).date() for x in legacy_records_};(ROOT/"legacy_training_only.json").write_text(json.dumps({"events":legacy_records_,"dates":sorted(map(str,legacy)),"policy":"training only; excluded plus/minus two days from all new systems"},indent=2))
 days=pd.date_range("2020-10-15","2025-12-31",freq="D",tz="UTC");days=[d for d in days if min(abs((d.date()-x).days) for x in legacy)>2]
 def one(d):
  o=daily_objects(d.to_pydatetime());sel=[x for x in o if 12<=_timestamp_from_key(x.key).hour or _timestamp_from_key(x.key).hour<4]
  return {"date":str(d.date()),"season":SEASONS[d.month],"year":d.year,"objects":len(sel),"mean_archive_bytes":float(np.mean([x.size for x in sel])) if sel else 0,"archive_bytes":sum(x.size for x in sel)}
 rows=[]
 with ThreadPoolExecutor(max_workers=16) as ex:
  fs={ex.submit(one,d):d for d in days}
  for i,f in enumerate(as_completed(fs),1):
   try:rows.append(f.result())
   except Exception as e:rows.append({"date":str(fs[f].date()),"error":str(e),"objects":0})
   if i%100==0:print(f"[inventory] {i}/{len(days)}",flush=True)
 all_=pd.DataFrame(rows).sort_values("date");all_.to_csv(ROOT/"archive_daily_inventory.csv",index=False)
 good=all_[all_.objects>=140].copy();chosen=[]
 # Balance year/season; archive byte size is only a source-presence proxy, never forecast skill.
 for (_, _),g in good.groupby(["year","season"]):
  for r in g.sort_values("mean_archive_bytes",ascending=False).itertuples(index=False):
   d=pd.Timestamp(r.date)
   if all(abs((d-pd.Timestamp(x.date)).days)>=3 for x in chosen):chosen.append(r)
   if sum(x.year==r.year and x.season==r.season for x in chosen)>=24:break
 pd.DataFrame(chosen).sort_values("date").to_csv(ROOT/"sparse_screen_candidates.csv",index=False);print(json.dumps({"archive_days":len(all_),"screen_candidates":len(chosen)},indent=2))
def screen():
 cand=pd.read_csv(ROOT/"sparse_screen_candidates.csv");raw=DATA/"screen";raw.mkdir(parents=True,exist_ok=True);manifest=DATA/"screen_downloads.jsonl";tasks=[]
 days=sorted({pd.Timestamp(d,tz="UTC") for d in cand.date}|{pd.Timestamp(d,tz="UTC")+pd.Timedelta(days=1) for d in cand.date})
 listings={}
 with ThreadPoolExecutor(max_workers=16) as ex:
  fs={ex.submit(daily_objects,d.to_pydatetime()):d for d in days}
  for i,f in enumerate(as_completed(fs),1):
   listings[fs[f]]=f.result()
   if i%100==0:print(f"[screen-list] {i}/{len(days)}",flush=True)
 for r in cand.itertuples(index=False):
  day=pd.Timestamp(r.date,tz="UTC");objs=listings[day]+listings[day+pd.Timedelta(days=1)];by={_timestamp_from_key(x.key):x for x in objs}
  for hour in [12,16,20,24]:
   t=day+pd.Timedelta(hours=hour);o=by.get(t.to_pydatetime());
   if o:tasks.append((r.date,t,o,raw/Path(o.key).name))
 def dl(x):
  date,t,o,p=x
  if not p.exists():download_immutable(object_url(BUCKET,o.key),p,manifest,metadata={"cycle2_phase":"sparse_screen","source_timestamp":t.isoformat()})
  return x
 with ThreadPoolExecutor(max_workers=16) as ex:list(ex.map(dl,tasks))
 out=[]
 with ProcessPoolExecutor(max_workers=8) as ex:
  for i,row in enumerate(ex.map(_screen_decode,tasks,chunksize=4),1):
   out.append(row)
   if i%100==0:print(f"[screen] {i}/{len(tasks)}",flush=True)
 s=pd.DataFrame(out);s.to_csv(ROOT/"sparse_screen_frames.csv",index=False);q=s.groupby("date",as_index=False).agg(screen_wet_fraction=("wet_fraction","max"),screen_heavy_fraction=("heavy_fraction","max"),screen_max_rate=("max_rate","max"),finite_fraction=("finite_fraction","min"));q["qualifies_sparse"]=q.screen_wet_fraction>=.001
 q=q.merge(cand[["date","season","year"]],on="date");q.to_csv(ROOT/"sparse_screen_summary.csv",index=False);print(json.dumps({"screened":len(q),"wet_candidates":int(q.qualifies_sparse.sum())},indent=2))
def materialize():
 q=pd.read_csv(ROOT/"sparse_screen_summary.csv");q=q[q.qualifies_sparse].sort_values(["season","screen_wet_fraction"],ascending=[True,False]);selected=[]
 quotas={"winter":18,"spring":24,"summer":30,"autumn":24}
 for season,g in q.groupby("season"):
  for r in g.itertuples(index=False):
   if all(abs((pd.Timestamp(r.date)-pd.Timestamp(x.date)).days)>=3 for x in selected):selected.append(r)
   if sum(x.season==season for x in selected)>=quotas[season]:break
 # Up to 96 supplies margin for the >=60 qualification gate.
 selected=sorted(selected,key=lambda x:x.screen_wet_fraction,reverse=True)[:96];pd.DataFrame(selected).to_csv(ROOT/"full_materialization_candidates.csv",index=False)
 cfg={"version":1,"projection":"EPSG:3978","sample_events":[{"id":f"cycle2_{str(r.date).replace('-','')}","start_utc":f"{r.date}T12:00:00+00:00","end_utc":str((pd.Timestamp(r.date,tz='UTC')+pd.Timedelta(days=1,hours=4)).isoformat()),"bbox_wgs84":BBOX} for r in selected]};cp=ROOT/"materialization_config.yaml";cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
 jobs=[(str(cp),e["id"],str(DATA/e["id"])) for e in cfg["sample_events"]]
 with ProcessPoolExecutor(max_workers=8) as ex:
  for i,result in enumerate(ex.map(_materialize_event,jobs),1):print(f"[materialize-complete] {i}/{len(jobs)} {result}",flush=True)
def _spread(rows, cap):
 rows=sorted(rows,key=lambda x:(x["issue_time_utc"],x["row_id"]))
 if len(rows)<=cap:return rows
 return [rows[i] for i in sorted(set(np.linspace(0,len(rows)-1,cap).round().astype(int)))]

def _stratified_split(inv):
 """Deterministic alternating greedy coverage over the complete inventory."""
 chosen={"new_dev":[],"new_final":[]};remaining=set(inv.system_id)
 lookup=inv.set_index("system_id")
 for _ in range(12):
  for split in ("new_dev","new_final"):
   counts={k:{} for k in ("season","year","regime","combo")}
   for sid in chosen[split]:
    r=lookup.loc[sid];vals={"season":r.season,"year":str(r.year),"regime":r.regime,"combo":f"{r.season}:{r.regime}"}
    for k,v in vals.items():counts[k][v]=counts[k].get(v,0)+1
   def score(sid):
    r=lookup.loc[sid];vals=(r.season,str(r.year),r.regime,f"{r.season}:{r.regime}")
    coverage=sum(w/(counts[k].get(v,0)+1) for k,v,w in zip(counts,vals,(8,3,6,2)))
    tie=int(hashlib.sha256(f"cycle2-split-v1:{split}:{sid}".encode()).hexdigest(),16)
    return coverage,-tie
   pick=max(remaining,key=score);chosen[split].append(pick);remaining.remove(pick)
 out=inv.copy();out["split"]="new_train"
 for split,ids in chosen.items():out.loc[out.system_id.isin(ids),"split"]=split
 return out

def _stratified_split_v2(inv):
 """Whole-inventory allocation with explicit TRAIN support constraints."""
 lookup=inv.set_index("system_id");all_ids=set(inv.system_id);reserved=set()
 def order(ids,label):return sorted(ids,key=lambda sid:hashlib.sha256(f"cycle2-split-v2:{label}:{sid}".encode()).hexdigest())
 # Reserve rare-regime TRAIN examples first, then minimum seasonal support.
 for regime,g in inv.groupby("regime"):
  if len(g)>=2:reserved.add(order(set(g.system_id),f"reserve-regime:{regime}")[0])
 for season,g in inv.groupby("season"):
  need=3 if season=="winter" and len(g)>=7 else 2 if len(g)>=6 else 1
  candidates=order(set(g.system_id),f"reserve-season:{season}")
  for sid in candidates:
   if len(set(g.system_id)&reserved)>=need:break
   reserved.add(sid)
 chosen={"new_dev":[],"new_final":[]};available=all_ids-reserved
 # Minimum evaluation season support; winter receives two each when possible.
 for split in ("new_dev","new_final"):
  for season,g in inv.groupby("season"):
   need=2 if season=="winter" and len(g)>=7 else 1 if len(g)>=6 else 0
   for sid in order(set(g.system_id)&available,f"{split}:season:{season}")[:need]:chosen[split].append(sid);available.remove(sid)
 # Fill alternately using descriptive coverage only.
 while len(chosen["new_dev"])<12 or len(chosen["new_final"])<12:
  for split in ("new_dev","new_final"):
   if len(chosen[split])>=12:continue
   counts={k:{} for k in ("season","year","regime","combo")}
   for sid in chosen[split]:
    r=lookup.loc[sid];vals={"season":r.season,"year":str(r.year),"regime":r.regime,"combo":f"{r.season}:{r.regime}"}
    for k,v in vals.items():counts[k][v]=counts[k].get(v,0)+1
   def score(sid):
    r=lookup.loc[sid];vals=(r.season,str(r.year),r.regime,f"{r.season}:{r.regime}");coverage=sum(w/(counts[k].get(v,0)+1) for k,v,w in zip(counts,vals,(8,3,6,2)));tie=int(hashlib.sha256(f"cycle2-split-v2:{split}:{sid}".encode()).hexdigest(),16);return coverage,-tie
   pick=max(available,key=score);chosen[split].append(pick);available.remove(pick)
 out=inv.copy();out["split"]="new_train"
 for split,ids in chosen.items():out.loc[out.system_id.isin(ids),"split"]=split
 return out

def repair_negatives():
 """Regenerate only hard negatives after the issue-frame validity amendment."""
 inv=pd.read_csv(ROOT/"frozen_system_split.csv");neg=[]
 for i,r in enumerate(inv.itertuples(index=False),1):
  with np.load(Path(r.source_path)) as z:
   rates=_downsample_max(z["rate_mm_hr"],2);times=pd.to_datetime(z["times"],utc=True)
  tiles={(0,0),(0,max(0,rates.shape[2]-128)),(max(0,rates.shape[1]-128),0),(max(0,rates.shape[1]-128),max(0,rates.shape[2]-128)),(max(0,(rates.shape[1]-128)//2),max(0,(rates.shape[2]-128)//2))};found=False
  for a in range(9,len(times)-20,20):
   for y0,x0 in sorted(tiles):
    w=rates[a-9:a+21,y0:y0+128,x0:x0+128];wf=np.nanmean(w>.1,axis=(1,2));valid=float(np.isfinite(w).mean());current_valid=float(np.isfinite(w[9]).mean())
    if valid>=.9 and current_valid>=.9 and wf.mean()<=.001 and wf.max()<=.005 and not np.any(w>=5):
     neg.append({"system_id":r.system_id,"row_id":f"{r.system_id}__dry__{times[a].strftime('%Y%m%dT%H%M')}","row_type":"hard_negative","issue_time_utc":times[a].isoformat(),"y0":y0,"y1":y0+127,"x0":x0,"x1":x0+127,"history_target_mean_wet_fraction":float(wf.mean()),"history_target_max_wet_fraction":float(wf.max()),"history_target_heavy_fraction":float(np.nanmean(w>=5)),"current_wet_fraction":float(wf[9]),"current_max_rate":float(np.nanmax(w[9])),"current_validity_fraction":current_valid,"validity_fraction":valid,"source_path":r.source_path,"source_sha256":r.source_sha256,"split":r.split});found=True;break
   if found:break
  if i%10==0:print(f"[negative-repair] {i}/{len(inv)} retained={len(neg)}",flush=True)
 nm=pd.DataFrame(neg);cm=pd.read_csv(ROOT/"clean_initiation_rows.csv");am=pd.read_csv(ROOT/"active_precip_rows.csv")
 nm[nm.split.ne("new_final")].to_csv(ROOT/"train_dev_hard_negatives.csv",index=False);nm[nm.split.eq("new_final")].drop(columns="source_path").to_csv(ROOT/"final_hard_negatives.csv",index=False)
 allrows=pd.concat([cm,am,nm],ignore_index=True,sort=False);allrows[allrows.split.ne("new_final")].to_csv(ROOT/"train_dev_rows.csv",index=False)
 finalrows=allrows[allrows.split.eq("new_final")];manifest={"status":"SEALED_UNSCORED","systems":inv[inv.split.eq("new_final")].drop(columns="source_path").to_dict("records"),"rows":finalrows.drop(columns="source_path").to_dict("records"),"row_type_counts":finalrows.row_type.value_counts().to_dict(),"source_hashes":dict(zip(inv[inv.split.eq("new_final")].system_id,inv[inv.split.eq("new_final")].source_sha256)),"model_predictions_generated":False,"future_targets_materialized_for_scoring":False};fp=ROOT/"final_manifest.json";fp.write_text(json.dumps(manifest,indent=2))
 report=json.loads((ROOT/"dataset_freeze_report.json").read_text());report["row_type_by_split"]=allrows.groupby(["split","row_type"]).size().unstack(fill_value=0).to_dict("index");report["final_manifest_sha256"]=sha256_file(fp);report["hard_negative_issue_frame_validity_amendment"]="regenerated before review; current frame finite fraction >=0.90";(ROOT/"dataset_freeze_report.json").write_text(json.dumps(report,indent=2));(ROOT/"split_freeze.json").write_text(json.dumps({k:report[k] for k in ("qualified_systems","system_split_counts","final_manifest_sha256","allocation")},indent=2));print(json.dumps({"hard_negatives":nm.split.value_counts().to_dict(),"final_manifest_sha256":report["final_manifest_sha256"]},indent=2))

def finalize_replacement_report():
 inv=pd.read_csv(ROOT/"frozen_system_split.csv");rows=pd.concat([pd.read_csv(ROOT/"clean_initiation_rows.csv"),pd.read_csv(ROOT/"active_precip_rows.csv"),pd.read_csv(ROOT/"train_dev_hard_negatives.csv"),pd.read_csv(ROOT/"final_hard_negatives.csv")],ignore_index=True,sort=False);audit=pd.read_csv(ROOT/"all_system_row_audit.csv");fp=ROOT/"final_manifest.json";superseded_hash="32d83c5c69e07cee1b5f072892500c05dbaf20679bb027b1d248b2555c64196e";superseded_reason="Residual-model qualification was still gated by clean-initiation count and produced an unsuitable training distribution.";old_rejected=set(pd.read_csv(ROOT/"superseded_initiation_gated_freeze"/"qualification_rejections.csv").system_id);recovered=len(old_rejected&set(inv.system_id));rejected=audit[audit.qualification_status.eq("rejected")]
 report={"status":"FROZEN_PRETRAINING_REPLACEMENT","qualification_rule":"integrity plus >=12 valid issue-time-selected active-precipitation rows before cap; clean initiation and hard-negative availability do not gate inclusion","materialized_systems":96,"qualified_systems":len(inv),"rejected_systems":len(rejected),"previously_rejected_recovered":recovered,"superseded_final_manifest_sha256":superseded_hash,"supersession_reason":superseded_reason,"rejection_breakdown":rejected.rejection_reason.value_counts().to_dict(),"rejected_system_details":rejected.to_dict("records"),"system_split_counts":inv.split.value_counts().to_dict(),"season_by_split":inv.groupby(["split","season"]).size().unstack(fill_value=0).to_dict("index"),"year_by_split":inv.groupby(["split","year"]).size().unstack(fill_value=0).to_dict("index"),"regime_by_split":inv.groupby(["split","regime"]).size().unstack(fill_value=0).to_dict("index"),"row_type_by_split":rows.groupby(["split","row_type"]).size().unstack(fill_value=0).to_dict("index"),"row_audit":audit.to_dict("records"),"source_hashes":dict(zip(inv.system_id,inv.source_sha256)),"final_manifest_sha256":sha256_file(fp),"allocation":"cycle2-split-v2 deterministic whole-inventory descriptive allocation with TRAIN season/regime reserves; no model metrics","legacy_events_unused_in_v1":True,"training_started":False,"cycle2_pysteps_forecasts_generated":False,"final_predictions_generated":False,"final_metrics_generated":False}
 (ROOT/"dataset_freeze_report.json").write_text(json.dumps(report,indent=2));(ROOT/"split_freeze.json").write_text(json.dumps({k:report[k] for k in ("qualified_systems","system_split_counts","final_manifest_sha256","allocation","superseded_final_manifest_sha256")},indent=2));print(json.dumps({k:report[k] for k in ("qualified_systems","rejected_systems","previously_rejected_recovered","system_split_counts","row_type_by_split","final_manifest_sha256")},indent=2))

def freeze():
 q=pd.read_csv(ROOT/"full_materialization_candidates.csv");systems=[];clean_rows=[];active_rows=[];neg=[];rejected=[];audit=[]
 for r in q.itertuples(index=False):
  eid=f"cycle2_{str(r.date).replace('-','')}";path=DATA/eid/"processed/events"/f"{eid}.npz"
  if not path.exists():rejected.append({"system_id":eid,"reason":"materialization_missing"});continue
  source_hash=sha256_file(path)
  try:
   z=np.load(path);native=z["rate_mm_hr"];times=pd.to_datetime(z["times"],utc=True);rates=_downsample_max(native,2)
   if len(times)!=161 or not bool(((times[1:]-times[:-1])==pd.Timedelta(minutes=6)).all()):raise ValueError("incomplete_exact_6min_window")
   lat=z["latitude"][:rates.shape[1]*2:2];lon=z["longitude"][:rates.shape[2]*2:2]
   cat=build_independent_initiation_catalog(rates,times,lat,lon,interval_minutes=6,resolution_km=2,history_minutes=60,horizon_minutes=120,rain_threshold=.1,strong_threshold=5,max_missing_fraction=.1,minimum_pixels=12)
   clean=cat[cat.apparent_initiation&~cat.advective_entry_like&~cat.touches_domain_boundary&(cat.future_max_mm_hr>=1)]
  except Exception as e:rejected.append({"system_id":eid,"reason":f"integrity:{e}"});continue
  maxr=float(np.nanmax(rates));wet=float(np.nanmean(rates>=.1));regime="convective" if maxr>=20 and wet<.08 else "stratiform" if maxr<10 and wet>=.02 else "mixed"
  system_record={"system_id":eid,"date":r.date,"year":int(str(r.date)[:4]),"start_utc":times[0].isoformat(),"end_utc":times[-1].isoformat(),"season":r.season,"regime":regime,"max_rate":maxr,"wet_fraction":wet,"clean_objects_available":len(clean),"source_path":path.as_posix(),"source_sha256":source_hash}
  cr=[]
  for x in clean.itertuples(index=False):
   issue=pd.Timestamp(x.representative_anchor_time);yc=(x.y_min+x.y_max)/2;xc=(x.x_min+x.x_max)/2;y0,y1=_tile_bounds(yc,128,rates.shape[1]);x0,x1=_tile_bounds(xc,128,rates.shape[2]);a=times.get_indexer([issue])[0]
   if a<9 or a+20>=len(times):continue
   valid=float(np.isfinite(rates[a-9:a+21,y0:y1+1,x0:x1+1]).mean())
   if valid>=.9:cr.append({"system_id":eid,"row_id":f"{eid}__clean__{x.event_id}","row_type":"clean_initiation","issue_time_utc":issue.isoformat(),"y0":y0,"y1":y1,"x0":x0,"x1":x1,"validity_fraction":valid,"source_path":path.as_posix(),"source_sha256":source_hash})
  clean_candidates=len(cr);cr=_spread(cr,12)
  ar=[]
  for a in range(9,len(times)-20,5):
   current=np.isfinite(rates[a])&(rates[a]>.1);labels,n=ndimage.label(current)
   comps=sorted(((int((labels==i).sum()),i) for i in range(1,n+1) if (labels==i).sum()>=12),reverse=True)[:2]
   for _,label_id in comps:
    yy,xx=np.where(labels==label_id);y0,y1=_tile_bounds(float(yy.mean()),128,rates.shape[1]);x0,x1=_tile_bounds(float(xx.mean()),128,rates.shape[2]);window=rates[a-9:a+21,y0:y1+1,x0:x1+1];valid=float(np.isfinite(window).mean())
    if valid<.9:continue
    issue=times[a];duplicate=any(x["issue_time_utc"]==issue.isoformat() and abs((x["y0"]+64)-(y0+64))<=32 and abs((x["x0"]+64)-(x0+64))<=32 for x in cr)
    if duplicate:continue
    tile=rates[a,y0:y1+1,x0:x1+1];ar.append({"system_id":eid,"row_id":f"{eid}__active__{issue.strftime('%Y%m%dT%H%M')}__{label_id}","row_type":"active_precip","selection_basis":"current_radar_only","issue_time_utc":issue.isoformat(),"y0":y0,"y1":y1,"x0":x0,"x1":x1,"current_wet_fraction":float(np.nanmean(tile>.1)),"current_max_rate":float(np.nanmax(tile)),"validity_fraction":valid,"source_path":path.as_posix(),"source_sha256":source_hash})
  active_candidates=len(ar);ar=_spread(ar,24)
  if active_candidates<12:
   reason="fewer_than_12_valid_active_precipitation_rows";rejected.append({"system_id":eid,"reason":reason,"active_candidates":active_candidates,"clean_objects":len(clean),"season":r.season,"year":int(str(r.date)[:4]),"regime":regime});audit.append({"system_id":eid,"season":r.season,"year":int(str(r.date)[:4]),"regime":regime,"active_candidates":active_candidates,"active_retained":len(ar),"clean_candidates":clean_candidates,"clean_retained":len(cr),"hard_negative_available":False,"qualification_status":"rejected","rejection_reason":reason});continue
  systems.append(system_record);clean_rows.extend(cr);active_rows.extend(ar)
  tiles={(0,0),(0,max(0,rates.shape[2]-128)),(max(0,rates.shape[1]-128),0),(max(0,rates.shape[1]-128),max(0,rates.shape[2]-128)),(max(0,(rates.shape[1]-128)//2),max(0,(rates.shape[2]-128)//2))};nr=[]
  for a in range(9,len(times)-20,20):
   for y0,x0 in sorted(tiles):
    w=rates[a-9:a+21,y0:y0+128,x0:x0+128];wf=np.nanmean(w>.1,axis=(1,2));valid=float(np.isfinite(w).mean());current_valid=float(np.isfinite(w[9]).mean())
    if valid>=.9 and current_valid>=.9 and wf.mean()<=.001 and wf.max()<=.005 and not np.any(w>=5):nr.append({"system_id":eid,"row_id":f"{eid}__dry__{times[a].strftime('%Y%m%dT%H%M')}","row_type":"hard_negative","issue_time_utc":times[a].isoformat(),"y0":y0,"y1":y0+127,"x0":x0,"x1":x0+127,"history_target_mean_wet_fraction":float(wf.mean()),"history_target_max_wet_fraction":float(wf.max()),"history_target_heavy_fraction":float(np.nanmean(w>=5)),"current_wet_fraction":float(wf[9]),"current_max_rate":float(np.nanmax(w[9])),"current_validity_fraction":current_valid,"validity_fraction":valid,"source_path":path.as_posix(),"source_sha256":source_hash});break
   if nr:break
  neg.extend(nr[:1]);audit.append({"system_id":eid,"season":r.season,"year":int(str(r.date)[:4]),"regime":regime,"active_candidates":active_candidates,"active_retained":len(ar),"clean_candidates":clean_candidates,"clean_retained":len(cr),"hard_negative_available":bool(nr),"qualification_status":"qualified","rejection_reason":""})
  if (len(systems)+len(rejected))%10==0:print(f"[freeze-audit] {len(systems)+len(rejected)}/{len(q)} qualified_so_far={len(systems)}",flush=True)
 pd.DataFrame(audit).to_csv(ROOT/"all_system_row_audit.csv",index=False);pd.DataFrame(rejected).to_csv(ROOT/"qualification_rejections.csv",index=False)
 if not systems:raise RuntimeError(f"No systems qualified; rejection breakdown: {pd.DataFrame(rejected).reason.value_counts().to_dict()}")
 inv=pd.DataFrame(systems).sort_values("date");inv.to_csv(ROOT/"qualified_system_inventory.csv",index=False)
 if len(inv)<60:raise RuntimeError(f"Only {len(inv)} genuinely independent systems qualified; stop before training")
 inv=_stratified_split_v2(inv)
 for season,g in inv.groupby("season"):
  counts=g.split.value_counts();required={"new_train":3,"new_dev":2,"new_final":2} if season=="winter" and len(g)>=7 else {"new_train":2,"new_dev":1,"new_final":1} if len(g)>=6 else {}
  if any(int(counts.get(k,0))<v for k,v in required.items()):raise RuntimeError(f"season support constraint failed for {season}: {counts.to_dict()}")
 for regime,g in inv.groupby("regime"):
  if len(g)>=2 and not (g.split=="new_train").any():raise RuntimeError(f"regime support constraint failed for {regime}")
 for split in ("new_dev","new_final"):
  part=inv[inv.split.eq(split)]
  if len(part)!=12 or part.season.nunique()<4 or part.regime.nunique()<min(3,inv.regime.nunique()) or part.year.nunique()<3:raise RuntimeError(f"degenerate {split} allocation")
 tables={"split_season":inv.groupby(["split","season"]).size(),"split_year":inv.groupby(["split","year"]).size(),"split_regime":inv.groupby(["split","regime"]).size(),"split_season_regime":inv.groupby(["split","season","regime"]).size()}
 for name,table in tables.items():print(f"\n[{name}]\n{table.to_string()}",flush=True)
 inv.to_csv(ROOT/"frozen_system_split.csv",index=False)
 (ROOT/"split_distribution.json").write_text(json.dumps({k:[dict(zip(v.index.names,idx if isinstance(idx,tuple) else (idx,)))|{"count":int(count)} for idx,count in v.items()] for k,v in tables.items()},indent=2))
 def attach(rows):return pd.DataFrame(rows).merge(inv[["system_id","split"]],on="system_id",validate="many_to_one")
 cm,am,nm=attach(clean_rows),attach(active_rows),attach(neg)
 cm.to_csv(ROOT/"clean_initiation_rows.csv",index=False);am.to_csv(ROOT/"active_precip_rows.csv",index=False)
 nm[nm.split.ne("new_final")].to_csv(ROOT/"train_dev_hard_negatives.csv",index=False);nm[nm.split.eq("new_final")].drop(columns="source_path").to_csv(ROOT/"final_hard_negatives.csv",index=False)
 nonfinal=pd.concat([cm,am,nm],ignore_index=True,sort=False);nonfinal[nonfinal.split.ne("new_final")].to_csv(ROOT/"train_dev_rows.csv",index=False)
 superseded_hash="32d83c5c69e07cee1b5f072892500c05dbaf20679bb027b1d248b2555c64196e";superseded_reason="Residual-model qualification was still gated by clean-initiation count and produced an unsuitable training distribution."
 finalrows=nonfinal[nonfinal.split.eq("new_final")];counts=finalrows.row_type.value_counts().to_dict();manifest={"status":"SEALED_UNSCORED","supersedes":{"status":"SUPERSEDED_PRETRAINING","final_manifest_sha256":superseded_hash,"reason":superseded_reason},"systems":inv[inv.split.eq("new_final")].drop(columns="source_path").to_dict("records"),"rows":finalrows.drop(columns="source_path").to_dict("records"),"row_type_counts":counts,"source_hashes":dict(zip(inv[inv.split.eq("new_final")].system_id,inv[inv.split.eq("new_final")].source_sha256)),"model_predictions_generated":False,"future_targets_materialized_for_scoring":False};fp=ROOT/"final_manifest.json";fp.write_text(json.dumps(manifest,indent=2))
 old_rejected=set(pd.read_csv(ROOT/"superseded_initiation_gated_freeze"/"qualification_rejections.csv").system_id);recovered=len(old_rejected&set(inv.system_id))
 rejected_frame=pd.DataFrame(rejected);rejection_breakdown=rejected_frame.reason.value_counts().to_dict() if "reason" in rejected_frame else {}
 report={"status":"FROZEN_PRETRAINING_REPLACEMENT","qualification_rule":"integrity plus >=12 valid issue-time-selected active-precipitation rows before cap; clean initiation and hard-negative availability do not gate inclusion","materialized_systems":int(sum((DATA/f"cycle2_{str(x.date).replace('-','')}"/"processed/events"/f"cycle2_{str(x.date).replace('-','')}.npz").exists() for x in q.itertuples())),"qualified_systems":len(inv),"rejected_systems":len(rejected),"previously_rejected_recovered":recovered,"superseded_final_manifest_sha256":superseded_hash,"supersession_reason":superseded_reason,"rejection_breakdown":rejection_breakdown,"rejected_system_details":rejected,"system_split_counts":inv.split.value_counts().to_dict(),"season_by_split":inv.groupby(["split","season"]).size().unstack(fill_value=0).to_dict("index"),"year_by_split":inv.groupby(["split","year"]).size().unstack(fill_value=0).to_dict("index"),"regime_by_split":inv.groupby(["split","regime"]).size().unstack(fill_value=0).to_dict("index"),"row_type_by_split":nonfinal.groupby(["split","row_type"]).size().unstack(fill_value=0).to_dict("index"),"row_audit":audit,"source_hashes":dict(zip(inv.system_id,inv.source_sha256)),"final_manifest_sha256":sha256_file(fp),"allocation":"cycle2-split-v2 deterministic whole-inventory descriptive allocation with TRAIN season/regime reserves; no model metrics","legacy_events_unused_in_v1":True,"training_started":False,"cycle2_pysteps_forecasts_generated":False,"final_predictions_generated":False,"final_metrics_generated":False}
 (ROOT/"dataset_freeze_report.json").write_text(json.dumps(report,indent=2));(ROOT/"split_freeze.json").write_text(json.dumps({k:report[k] for k in ("qualified_systems","system_split_counts","final_manifest_sha256","allocation")},indent=2))
 doc=Path("docs/cycle2_dataset_freeze.md");doc.write_text("# Cycle 2 dataset freeze\n\n**Status:** FROZEN PRE-TRAINING; model training is not authorized by this pass.\n\n"+f"- Materialized systems: {report['materialized_systems']}\n- Qualified independent systems: {report['qualified_systems']}\n- Rejected systems: {report['rejected_systems']}\n- System splits: `{report['system_split_counts']}`\n- Row types by split: `{report['row_type_by_split']}`\n- Final manifest SHA-256: `{report['final_manifest_sha256']}`\n- Legacy events are recorded but unused in V1.\n\n## Distributions\n\n"+f"Season: `{report['season_by_split']}`\n\nYear: `{report['year_by_split']}`\n\nRegime: `{report['regime_by_split']}`\n\n## Selection and sealing\n\nThe allocation uses deterministic alternating greedy coverage across the entire qualified inventory using season, year, regime, and season-regime only. Clean-initiation rows are capped at 12, issue-time-only active-precipitation rows at 24, and hard negatives at one per system. NEW FINAL remains `SEALED_UNSCORED`; no PySTEPS forecasts, learned inputs, predictions, or metrics were generated for it. Full per-system row audits, rejection reasons, and source hashes are in `artifacts/cycle2/data/dataset_freeze_report.json`.\n",encoding="utf-8")
 print(json.dumps({k:report[k] for k in ("materialized_systems","qualified_systems","rejected_systems","system_split_counts","row_type_by_split","final_manifest_sha256")},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("phase",choices=["inventory","screen","materialize","freeze","repair_negatives","finalize_replacement_report"]);a=p.parse_args();globals()[a.phase]()
if __name__=="__main__":main()
