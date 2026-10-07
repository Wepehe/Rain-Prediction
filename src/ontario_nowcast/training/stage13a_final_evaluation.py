"""One-time final evaluation of the frozen Stage 13A procedure."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata
from ..data.manifest import sha256_file
from ..models.baselines import pysteps_deterministic_extrapolation,pysteps_steps_ensemble
from ..sample_evaluation import _downsample_max
from .stage13_probabilistic_pysteps import _event_path,_seed

ROOT=Path("artifacts/stage_13a/final_evaluation"); PROC=Path("artifacts/stage_13a/final_procedure_manifest.json"); POS=Path("artifacts/stage_8/qualification/stage8_final_object_manifest.csv"); NEG=Path("artifacts/stage_8/qualification/stage8_hard_negative_manifest.csv"); FREEZE=Path("artifacts/stage_8/qualification/stage8_qualification_freeze.json"); THR=[.1,1.,2.5,5.]
def first(a):
 h=a[..., :-1,:,:]&a[...,1:,:,:] if a.ndim==4 else a[:-1]&a[1:];o=np.full(a.shape[-2:],126,dtype=np.int16) if a.ndim==3 else np.full((a.shape[0],*a.shape[-2:]),126,dtype=np.int16);ok=h.any(-3);o[ok]=(h.argmax(-3)[ok]+1)*6;return o
def score(y,q):
 tp=np.sum(y&q);fp=np.sum(~y&q);fn=np.sum(y&~q);p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);return {"precision":p,"pod":r,"far":fp/max(tp+fp,1),"f1":2*p*r/max(p+r,1e-12),"csi":tp/max(tp+fp+fn,1)}
def fss(y,q,n):
 size=(1,n,n);a=uniform_filter(y.astype(np.float32),size,mode="constant");b=uniform_filter(q.astype(np.float32),size,mode="constant");return 1-np.mean((a-b)**2)/max(np.mean(a*a+b*b),1e-12)
def auc(y,p):
 y=np.asarray(y,bool);n1=y.sum();n0=(~y).sum();return float((rankdata(p)[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else np.nan
def verify():
 proc=json.loads(PROC.read_text());pre=json.loads(Path("artifacts/stage_13/stage13_predeclaration.json").read_text());freeze=json.loads(FREEZE.read_text())
 checks={"procedure_sha256":sha256_file(PROC),"stage13_predeclaration_sha256":sha256_file(Path("artifacts/stage_13/stage13_predeclaration.json")),"code_hash_matches":sha256_file(Path(proc["evaluation_code"]))==proc["evaluation_code_sha256"],"ensemble_manifest_hash_matches":sha256_file(Path("artifacts/stage_13/ensemble_manifest.csv"))==proc["ensemble_manifest_sha256"],"decision_hash_matches":sha256_file(Path("artifacts/stage_13a/stage13a_decision.json"))==proc["stage13a_decision_sha256"],"final_manifest_hash_matches":sha256_file(POS)==freeze["files"]["final_objects"]["sha256"],"hard_negative_manifest_hash_matches":sha256_file(NEG)==freeze["files"]["hard_negatives"]["sha256"]}
 required=pre["pysteps_version"]=="1.21.5" and pre["ensemble"]=={"method":"STEPS","members":8,"cascade_levels":"PySTEPS default (6)","ar_order":"PySTEPS default (2)","noise_method":"PySTEPS default nonparametric","velocity_perturbations":"PySTEPS default","motion":"LK","input_frames":10,"km_per_pixel":2.0,"timestep_minutes":6,"rain_threshold_mm_hr":0.1,"transform":"dB","seed_policy":"uint32(first 8 hex SHA256(row_id + ':stage13:v1'))","num_workers":1,"fft_method":"numpy"} and proc["deployment_probability_threshold"]==.125 and proc["persistence"]=="two consecutive six-minute frames"
 checks["exact_configuration_matches"]=required
 p=pd.read_csv(POS);n=pd.read_csv(NEG);n=n[n.split.eq("final")];checks.update({"positive_systems":p.independent_system_group.nunique(),"positive_rows":len(p),"unique_issue_times":p.issue_time_utc.nunique(),"hard_negative_rows":len(n),"hard_negative_systems":n.independent_system_group.nunique(),"expected_counts_match":p.independent_system_group.nunique()==6 and len(p)==142 and p.issue_time_utc.nunique()==124 and len(n)==8 and n.independent_system_group.nunique()==4})
 if not all(v for k,v in checks.items() if isinstance(v,bool)):raise RuntimeError(checks)
 ROOT.mkdir(parents=True,exist_ok=True);(ROOT/"preforecast_integrity.json").write_text(json.dumps(checks,indent=2));return p,n,checks
def failure(py,y,rate,target):
 inter=(py&y).sum();union=(py|y).sum();iou=inter/union if union else 1
 if y.any() and not py.any():return "no_initiation_signal"
 if py.mean()>y.mean()*1.5:return "decay_persistence"
 if y.mean()>py.mean()*1.5:return "growth_underprediction"
 if y.any() and py.any() and iou<.2:return "displacement"
 if np.nanmean(np.abs(rate-target))>1:return "intensity_error"
 return "other_or_adequate"
def run():
 p,n,integrity=verify();p=p.rename(columns={"object_id":"row_id"});p["role"]="positive";n=n.rename(columns={"hard_negative_id":"row_id"});n["role"]="hard_negative";n["radar_source_path"]=None;cols=["row_id","role","event_id","independent_system_group","issue_time_utc","tile_y_min","tile_y_max","tile_x_min","tile_x_max","radar_source_path"];rows=pd.concat([p[cols],n[cols]],ignore_index=True)
 evcache={};metrics=[];prob=[];timing=[];hard=[];spread=[];blind=[];sources=[]
 for i,r in enumerate(rows.itertuples(index=False),1):
  if r.event_id not in evcache:
   path=_event_path(r.event_id,r.radar_source_path);z=np.load(path);evcache[r.event_id]=(path,_downsample_max(z["rate_mm_hr"],2),pd.to_datetime(z["times"],utc=True));sources.append({"event_id":r.event_id,"path":path.as_posix(),"sha256":sha256_file(path)})
  path,rate,times=evcache[r.event_id];ix=times.get_indexer([pd.Timestamp(r.issue_time_utc)])[0];y0,y1,x0,x1=map(int,[r.tile_y_min,r.tile_y_max,r.tile_x_min,r.tile_x_max]);hist=rate[ix-9:ix+1,y0:y1+1,x0:x1+1];truth=rate[ix+1:ix+21,y0:y1+1,x0:x1+1];valid=np.isfinite(truth)
  if hist.shape!=(10,128,128) or truth.shape!=(20,128,128):raise RuntimeError(f"shape failure {r.row_id}")
  det=pysteps_deterministic_extrapolation(hist[-3:],20);ens=pysteps_steps_ensemble(hist,20,km_per_pixel=2,timestep_minutes=6,ensemble_members=8,rain_threshold=.1,seed=_seed(r.row_id));pr=np.stack([(ens>x).mean(0) for x in THR],0);mean=ens.mean(0);sd=ens.std(0)
  if np.any(np.diff(pr,axis=0)>0):raise RuntimeError(f"monotonicity {r.row_id}")
  yt=(truth>.1)&valid;policies={"deterministic":det>.1,"fixed_0.5":pr[0]>=.5,"stage13a_0.125":pr[0]>=.125};ot=first(yt)
  for name,q in policies.items():
   s=score(yt[valid],q[valid]);qt=first(q);active=ot<=120;s.update({"row_id":r.row_id,"role":r.role,"event":r.independent_system_group,"policy":name,"fss6":fss(yt,q,3),"fss18":fss(yt,q,9),"fss36":fss(yt,q,18),"onset_mae":float(np.mean(np.abs(qt[active]-ot[active]))) if active.any() else np.nan,"onset_bias":float(np.mean(qt[active]-ot[active])) if active.any() else np.nan});metrics.append(s)
  mt=first(ens>.1);active=ot<=120
  if active.any():
   lo=np.quantile(mt,.125,axis=0);med=np.median(mt,axis=0);hi=np.quantile(mt,.875,axis=0);v=mt[:,active];timing.append({"row_id":r.row_id,"event":r.independent_system_group,"kind":"onset","median_mae":float(np.mean(np.abs(med[active]-ot[active]))),"median_bias":float(np.mean(med[active]-ot[active])),"coverage":float(np.mean((ot[active]>=lo[active])&(ot[active]<=hi[active]))),"width":float(np.mean(hi[active]-lo[active])),"no_member_fraction":float(np.mean((v>120).all(0))),"by30":float(np.mean(v<=30)),"by60":float(np.mean(v<=60)),"by90":float(np.mean(v<=90)),"by120":float(np.mean(v<=120)),"median":float(np.mean(med[active]))})
  ca=yt[0]&valid.all(0);co=first(~yt);ct=first(~(ens>.1));cd=first(~(det>.1))
  if ca.any():
   lo=np.quantile(ct,.125,axis=0);med=np.median(ct,axis=0);hi=np.quantile(ct,.875,axis=0);timing.append({"row_id":r.row_id,"event":r.independent_system_group,"kind":"cessation","median_mae":float(np.mean(np.abs(med[ca]-co[ca]))),"deterministic_mae":float(np.mean(np.abs(cd[ca]-co[ca]))),"median_bias":float(np.mean(med[ca]-co[ca])),"coverage":float(np.mean((co[ca]>=lo[ca])&(co[ca]<=hi[ca]))),"width":float(np.mean(hi[ca]-lo[ca]))})
  for j in range(20):
   cat=failure(det[j]>.1,truth[j]>.1,det[j],truth[j]);err=float(np.nanmean(np.abs(mean[j]-truth[j])));spr=float(np.nanmean(sd[j]));spread.append({"row_id":r.row_id,"event":r.independent_system_group,"lead_minutes":6*(j+1),"category":cat,"spread":spr,"abs_error":err})
   if cat=="no_initiation_signal":c=(ens[:,j]>.1).sum(0);blind.append({"row_id":r.row_id,"event":r.independent_system_group,"zero":float(np.mean(c==0)),"one":float(np.mean(c>=1)),"four":float(np.mean(c>=4))})
   for ti,t in enumerate(THR):
    v=valid[j]&np.isfinite(pr[ti,j]);ind=np.flatnonzero(v.ravel())[::64];yy=(truth[j].ravel()[ind]>t);pp=pr[ti,j].ravel()[ind];dd=det[j].ravel()[ind]>t;prob.extend({"event":r.independent_system_group,"role":r.role,"threshold":t,"lead_group":["0-30","30-60","60-90","90-120"][min(j//5,3)],"y":int(a),"p":float(b),"det":int(c)} for a,b,c in zip(yy,pp,dd))
  if r.role=="hard_negative":hard.append({"row_id":r.row_id,"event":r.independent_system_group,"mean_probability":float(pr[0].mean()),"max_probability":float(pr[0].max()),"wet_area":float((pr[0]>=.125).mean()),"false_initiation":float((pr[0]>=.125).any()),"brier":float(np.mean(pr[0]**2))})
  if i%10==0:print(f"[stage13a-final] {i}/{len(rows)}",flush=True)
 md=pd.DataFrame(metrics);pd.DataFrame(sources).drop_duplicates().to_csv(ROOT/"radar_source_manifest.csv",index=False);md.to_csv(ROOT/"policy_metrics_by_row.csv",index=False);pd.DataFrame(timing).to_csv(ROOT/"exact_timing_by_row.csv",index=False);pd.DataFrame(hard).to_csv(ROOT/"hard_negative_by_row.csv",index=False);pd.DataFrame(spread).to_csv(ROOT/"spread_error.csv",index=False);pd.DataFrame(blind).to_csv(ROOT/"radar_blind_behavior.csv",index=False)
 event=md[md.role.eq("positive")].groupby(["event","policy"],as_index=False).mean(numeric_only=True);event.to_csv(ROOT/"policy_metrics_by_event.csv",index=False);macro=event.groupby("policy",as_index=False).mean(numeric_only=True);macro.to_csv(ROOT/"policy_metrics_event_macro.csv",index=False)
 pdx=pd.DataFrame(prob);reli=pdx.assign(bin=pd.cut(pdx.p,np.linspace(0,1,11),include_lowest=True)).groupby(["threshold","lead_group","bin"],observed=False).agg(samples=("y","size"),mean_probability=("p","mean"),observed_frequency=("y","mean")).reset_index();reli.to_csv(ROOT/"reliability.csv",index=False)
 pe=[]
 for (e,t,l),g in pdx.groupby(["event","threshold","lead_group"]):pe.append({"event":e,"threshold":t,"lead_group":l,"brier":float(np.mean((g.p-g.y)**2)),"deterministic_brier":float(np.mean((g.det-g.y)**2)),"auc":auc(g.y,g.p),"sharpness":float(g.p.std())})
 pe=pd.DataFrame(pe);pe.to_csv(ROOT/"probability_metrics_by_event.csv",index=False);pm=pe.groupby(["threshold","lead_group"],as_index=False).mean(numeric_only=True);pm.to_csv(ROOT/"probability_metrics_event_macro.csv",index=False)
 tm=pd.DataFrame(timing);on=tm[tm.kind.eq("onset")].groupby("event").mean(numeric_only=True).mean();ce=tm[tm.kind.eq("cessation")].groupby("event").mean(numeric_only=True).mean();hd=pd.DataFrame(hard);hde=hd.groupby("event",as_index=False).mean(numeric_only=True);hde.to_csv(ROOT/"hard_negative_by_event.csv",index=False);sp=pd.DataFrame(spread);corr={"overall":sp.spread.corr(sp.abs_error)}
 for c in ["displacement","decay_persistence","growth_underprediction","no_initiation_signal"]:corr[c]=sp[sp.category.eq(c)].spread.corr(sp[sp.category.eq(c)].abs_error)
 bm=pd.DataFrame(blind).mean(numeric_only=True).to_dict(); mm=macro.set_index("policy");det=mm.loc["deterministic"];s13=mm.loc["fixed_0.5"];a=mm.loc["stage13a_0.125"];evp=event.pivot(index="event",columns="policy",values="f1");fdelta=evp["stage13a_0.125"]-evp["deterministic"];be=pe.groupby("event")[["brier","deterministic_brier"]].mean();bwins=int((be.brier<be.deterministic_brier).sum())
 classification="FINAL VALIDATED — PROBABILISTIC PYSTEPS OPERATIONAL BASELINE" if a.f1>det.f1 and a.fss18>=det.fss18-.01 and bwins>=4 and hd.wet_area.mean()<=.01 else "FINAL MIXED — DEVELOPMENT BENEFIT DID NOT CLEANLY GENERALIZE" if a.f1>=det.f1-.005 else "FINAL NEGATIVE — DEVELOPMENT BENEFIT FAILED TO GENERALIZE";tclass="TIMING UNCERTAINTY REMAINS UNDER-DISPERSED" if on.coverage<.7 else "TIMING UNCERTAINTY IMPROVED BUT NOT CALIBRATED" if on.coverage<.8 else "TIMING UNCERTAINTY USEFUL"
 result={"final_set_manifest_sha256":sha256_file(POS),"procedure_manifest_sha256":sha256_file(PROC),"systems":6,"positive_rows":142,"hard_negative_rows":8,"macro_metrics":{x:{k:float(mm.loc[x,k]) for k in ["csi","pod","far","f1","fss6","fss18","fss36","onset_mae","onset_bias"]} for x in mm.index},"event_metrics":event.to_dict("records"),"probability_metrics":pm.to_dict("records"),"brier_systems_improved":bwins,"systems_f1_improved":int((fdelta>.005).sum()),"systems_f1_approximately_tied":int((fdelta.abs()<=.005).sum()),"systems_f1_worsened":int((fdelta<-.005).sum()),"exact_onset":{k:float(on[k]) for k in ["median_mae","median_bias","coverage","width","no_member_fraction","by30","by60","by90","by120","median"]},"exact_cessation":{k:float(ce[k]) for k in ["median_mae","deterministic_mae","median_bias","coverage","width"]},"radar_blind_behavior":bm,"spread_error_correlation":corr,"hard_negative_macro":hd.mean(numeric_only=True).to_dict(),"hard_negative_events":hde.to_dict("records"),"classification":classification,"timing_classification":tclass,"sealed_final_now_permanently_consumed":True,"post_final_retuning_permitted":False}
 result["artifact_hashes"]={p.name:sha256_file(p) for p in ROOT.iterdir() if p.is_file() and p.name not in ["stage13a_final_evaluation.json","artifact_hashes.json"]};(ROOT/"stage13a_final_evaluation.json").write_text(json.dumps(result,indent=2,allow_nan=False));files={p.name:sha256_file(p) for p in ROOT.iterdir() if p.is_file() and p.name!="artifact_hashes.json"};(ROOT/"artifact_hashes.json").write_text(json.dumps(files,indent=2));print(json.dumps({k:result[k] for k in ["classification","timing_classification","brier_systems_improved","systems_f1_improved","systems_f1_approximately_tied","systems_f1_worsened"]},indent=2))
def summarize():
 event=pd.read_csv(ROOT/"policy_metrics_by_event.csv");macro=pd.read_csv(ROOT/"policy_metrics_event_macro.csv");pm=pd.read_csv(ROOT/"probability_metrics_event_macro.csv");pe=pd.read_csv(ROOT/"probability_metrics_by_event.csv");tm=pd.read_csv(ROOT/"exact_timing_by_row.csv");hd=pd.read_csv(ROOT/"hard_negative_by_row.csv");hde=pd.read_csv(ROOT/"hard_negative_by_event.csv");sp=pd.read_csv(ROOT/"spread_error.csv");bl=pd.read_csv(ROOT/"radar_blind_behavior.csv")
 on=tm[tm.kind.eq("onset")].groupby("event").mean(numeric_only=True).mean();ce=tm[tm.kind.eq("cessation")].groupby("event").mean(numeric_only=True).mean();corr={"overall":sp.spread.corr(sp.abs_error)}
 for c in ["displacement","decay_persistence","growth_underprediction","no_initiation_signal"]:corr[c]=sp[sp.category.eq(c)].spread.corr(sp[sp.category.eq(c)].abs_error)
 bm=bl.mean(numeric_only=True).to_dict();mm=macro.set_index("policy");det=mm.loc["deterministic"];a=mm.loc["stage13a_0.125"];evp=event.pivot(index="event",columns="policy",values="f1");fdelta=evp["stage13a_0.125"]-evp["deterministic"];be=pe.groupby("event")[["brier","deterministic_brier"]].mean();bwins=int((be.brier<be.deterministic_brier).sum())
 classification="FINAL VALIDATED — PROBABILISTIC PYSTEPS OPERATIONAL BASELINE" if a.f1>det.f1 and a.fss18>=det.fss18-.01 and bwins>=4 and hd.wet_area.mean()<=.01 else "FINAL MIXED — DEVELOPMENT BENEFIT DID NOT CLEANLY GENERALIZE" if a.f1>=det.f1-.005 else "FINAL NEGATIVE — DEVELOPMENT BENEFIT FAILED TO GENERALIZE";tclass="TIMING UNCERTAINTY REMAINS UNDER-DISPERSED" if on.coverage<.7 else "TIMING UNCERTAINTY IMPROVED BUT NOT CALIBRATED" if on.coverage<.8 else "TIMING UNCERTAINTY USEFUL"
 result={"final_set_manifest_sha256":sha256_file(POS),"procedure_manifest_sha256":sha256_file(PROC),"systems":6,"positive_rows":142,"hard_negative_rows":8,"macro_metrics":{x:{k:float(mm.loc[x,k]) for k in ["csi","pod","far","f1","fss6","fss18","fss36","onset_mae","onset_bias"]} for x in mm.index},"event_metrics":event.to_dict("records"),"probability_metrics":pm.to_dict("records"),"brier_systems_improved":bwins,"systems_f1_improved":int((fdelta>.005).sum()),"systems_f1_approximately_tied":int((fdelta.abs()<=.005).sum()),"systems_f1_worsened":int((fdelta<-.005).sum()),"exact_onset":{k:float(on[k]) for k in ["median_mae","median_bias","coverage","width","no_member_fraction","by30","by60","by90","by120","median"]},"exact_cessation":{k:float(ce[k]) for k in ["median_mae","deterministic_mae","median_bias","coverage","width"]},"radar_blind_behavior":bm,"spread_error_correlation":corr,"hard_negative_macro":hd.mean(numeric_only=True).to_dict(),"hard_negative_events":hde.to_dict("records"),"classification":classification,"timing_classification":tclass,"sealed_final_now_permanently_consumed":True,"post_final_retuning_permitted":False}
 result["artifact_hashes"]={p.name:sha256_file(p) for p in ROOT.iterdir() if p.is_file() and p.name not in ["stage13a_final_evaluation.json","artifact_hashes.json"]};(ROOT/"stage13a_final_evaluation.json").write_text(json.dumps(result,indent=2,allow_nan=False));files={p.name:sha256_file(p) for p in ROOT.iterdir() if p.is_file() and p.name!="artifact_hashes.json"};(ROOT/"artifact_hashes.json").write_text(json.dumps(files,indent=2));print(json.dumps({k:result[k] for k in ["classification","timing_classification","brier_systems_improved","systems_f1_improved","systems_f1_approximately_tied","systems_f1_worsened"]},indent=2))
if __name__=="__main__":
 import sys
 summarize() if len(sys.argv)>1 and sys.argv[1]=="summarize" else run()
