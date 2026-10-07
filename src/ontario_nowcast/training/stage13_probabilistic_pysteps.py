"""Frozen Stage 13 probabilistic PySTEPS development experiment."""
from __future__ import annotations

import hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.special import ndtr
from scipy.stats import rankdata

from ..data.manifest import sha256_file
from ..models.baselines import pysteps_steps_ensemble
from ..sample_evaluation import _downsample_max

ROOT=Path("artifacts/stage_13"); SRC=ROOT/"ensemble_cache"
THR=np.array([.1,1.,2.5,5.],dtype=np.float32); LEADS=[5,10,15,20]
CONFIG={"schema_version":1,"status":"FROZEN_BEFORE_SKILL_EVALUATION","reference":{"method":"extrapolation","motion":"LK","input_frames":3,"transform":"dB","zerovalue":-15.0,"rain_threshold_mm_hr":.1,"timestep_minutes":6,"lead_steps":20,"output_units":"mm/h","stochastic":False},"ensemble":{"method":"STEPS","members":8,"cascade_levels":"PySTEPS default (6)","ar_order":"PySTEPS default (2)","noise_method":"PySTEPS default nonparametric","velocity_perturbations":"PySTEPS default","motion":"LK","input_frames":10,"km_per_pixel":2.0,"timestep_minutes":6,"rain_threshold_mm_hr":.1,"transform":"dB","seed_policy":"uint32(first 8 hex SHA256(row_id + ':stage13:v1'))","num_workers":1,"fft_method":"numpy"},"probability_thresholds_mm_hr":THR.tolist(),"categorical_policy":"ensemble exceedance probability >= 0.5","persistent_wet_definition":"two consecutive six-minute frames >= threshold","neighborhood_km":[6,18,36],"calibration_policy":"diagnose raw only; no calibrator unless discrimination is useful and bias systematic","sealed_final_accessed":False}

def _seed(rid): return int(hashlib.sha256(f"{rid}:stage13:v1".encode()).hexdigest()[:8],16)
def _event_path(eid,source=None):
 p=Path(source) if isinstance(source,str) else Path("_")
 choices=[p,Path(f"data/stage8_qualification/{eid}/processed/events/{eid}.npz"),Path(f"data/s8/e01/processed/events/{eid}.npz"),Path(f"data/s8/e02/processed/events/{eid}.npz")]
 return next(x for x in choices if x.exists())
def rows():
 p=pd.read_csv("artifacts/stage_8/qualification/stage8_development_object_manifest.csv").rename(columns={"object_id":"row_id"});p["role"]="positive"
 n=pd.read_csv("artifacts/stage_8/qualification/stage8_hard_negative_manifest.csv");n=n[n.split.eq("development")].rename(columns={"hard_negative_id":"row_id"});n["role"]="hard_negative";n["radar_source_path"]=None
 c=["row_id","role","event_id","independent_system_group","issue_time_utc","tile_y_min","tile_y_max","tile_x_min","tile_x_max","radar_source_path"]
 return pd.concat([p[c],n[c]],ignore_index=True)
def predeclare():
 ROOT.mkdir(parents=True,exist_ok=True); from importlib.metadata import version
 spec=dict(CONFIG);spec["pysteps_version"]=version("pysteps");spec["rows"]=len(rows());spec["positive_rows"]=int((rows().role=="positive").sum());spec["hard_negative_rows"]=int((rows().role=="hard_negative").sum());spec["systems"]=int(rows().independent_system_group.nunique())
 path=ROOT/"stage13_predeclaration.json";path.write_text(json.dumps(spec,indent=2));print(json.dumps({"path":str(path),"sha256":sha256_file(path)},indent=2))
def materialize():
 spec=json.loads((ROOT/"stage13_predeclaration.json").read_text()); assert not spec["sealed_final_accessed"];SRC.mkdir(parents=True,exist_ok=True)
 rr=rows(); ev={}; out=[]
 for i,r in enumerate(rr.itertuples(index=False),1):
  if r.event_id not in ev:
   z=np.load(_event_path(r.event_id,r.radar_source_path));ev[r.event_id]=(_downsample_max(z["rate_mm_hr"],2),pd.to_datetime(z["times"],utc=True))
  rate,times=ev[r.event_id];ix=times.get_indexer([pd.Timestamp(r.issue_time_utc)])[0];y=slice(int(r.tile_y_min),int(r.tile_y_max)+1);x=slice(int(r.tile_x_min),int(r.tile_x_max)+1);hist=rate[ix-9:ix+1,y,x]
  ens=pysteps_steps_ensemble(hist,20,km_per_pixel=2.,timestep_minutes=6.,ensemble_members=8,rain_threshold=.1,seed=_seed(r.row_id))
  probs=np.stack([(ens>t).mean(0) for t in THR],0).astype(np.float16); mean=ens.mean(0).astype(np.float16);spread=ens.std(0).astype(np.float16)
  if np.any(np.diff(probs.astype(float),axis=0)>1e-7): raise RuntimeError(f"probability monotonicity violation: {r.row_id}")
  path=SRC/f"{hashlib.sha256(r.row_id.encode()).hexdigest()[:20]}.npz";np.savez_compressed(path,probability=probs,ensemble_mean=mean,ensemble_spread=spread)
  out.append({"row_id":r.row_id,"role":r.role,"event_id":r.event_id,"independent_system_group":r.independent_system_group,"issue_time_utc":r.issue_time_utc,"seed":_seed(r.row_id),"cache_path":path.as_posix(),"sha256":sha256_file(path),"monotonic":True})
  if i%10==0: print(f"[stage13] {i}/{len(rr)}",flush=True)
 pd.DataFrame(out).to_csv(ROOT/"ensemble_manifest.csv",index=False)
 # Re-run one row to make the fixed seed test independently auditable.
 first=rr.iloc[0];rate,times=ev[first.event_id];ix=times.get_indexer([pd.Timestamp(first.issue_time_utc)])[0];y=slice(int(first.tile_y_min),int(first.tile_y_max)+1);x=slice(int(first.tile_x_min),int(first.tile_x_max)+1)
 a=pysteps_steps_ensemble(rate[ix-9:ix+1,y,x],20,km_per_pixel=2.,timestep_minutes=6.,ensemble_members=8,rain_threshold=.1,seed=_seed(first.row_id)); b=pysteps_steps_ensemble(rate[ix-9:ix+1,y,x],20,km_per_pixel=2.,timestep_minutes=6.,ensemble_members=8,rain_threshold=.1,seed=_seed(first.row_id))
 finite=np.isfinite(a)&np.isfinite(b); exact=bool(np.array_equal(a,b,equal_nan=True)); delta=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
 (ROOT/"reproducibility.json").write_text(json.dumps({"row_id":first.row_id,"exact_repeat":exact,"identical_nan_mask":bool(np.array_equal(np.isnan(a),np.isnan(b))),"max_abs_difference_finite":delta},indent=2))
def _auc(y,p):
 y=np.asarray(y,dtype=bool);p=np.asarray(p);n1=y.sum();n0=(~y).sum()
 return float((rankdata(p)[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else np.nan
def _scores(y,p,cut=.5):
 q=p>=cut;tp=np.sum(q&y);fp=np.sum(q&~y);fn=np.sum(~q&y);prec=tp/max(tp+fp,1);rec=tp/max(tp+fn,1)
 return prec,rec,2*prec*rec/max(prec+rec,1e-12),tp/max(tp+fp+fn,1),fp/max(tp+fp,1)
def _fss(y,q,n):
 a=uniform_filter(y.astype(float),size=n,mode="constant");b=uniform_filter(q.astype(float),size=n,mode="constant");return 1-float(np.mean((a-b)**2))/max(float(np.mean(a*a+b*b)),1e-12)
def _first_persistent(a):
 hit=a[:-1]&a[1:];out=np.full(a.shape[1:],126,dtype=np.int16);anyh=hit.any(0);out[anyh]=(hit.argmax(0)[anyh]+1)*6;return out
def _quantile_time_from_marginals(p,q,wet=True):
 # Conservative two-frame-persistence approximation from marginal probabilities.
 x=np.minimum(p[:-1],p[1:]) if wet else np.minimum(1-p[:-1],1-p[1:]);cdf=np.maximum.accumulate(x,axis=0);hit=cdf>=q;out=np.full(p.shape[1:],126,dtype=np.int16);ok=hit.any(0);out[ok]=(hit.argmax(0)[ok]+1)*6;return out
def evaluate():
 man=pd.read_csv(ROOT/"ensemble_manifest.csv"); base=pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv").set_index("row_id")
 failure=pd.read_csv("artifacts/stage_9/source_advantage/oracle_source_advantage_by_row_lead.csv");failure=failure[failure.collection.eq("stage8_nested_development")][["row_id","lead_minutes","pysteps_failure_category"]]
 rec=[];samples=[];onsets=[];cessations=[];hard=[];spreadcat=[]
 for r in man.itertuples(index=False):
  e=np.load(r.cache_path);z=np.load(base.loc[r.row_id,"cache_path"]);pr=e["probability"].astype(float);em=e["ensemble_mean"].astype(float);sp=e["ensemble_spread"].astype(float);det=z["pysteps_rate"].astype(float);truth=z["target_rate"].astype(float);valid=z["target_valid"].astype(bool)
  for ti,t in enumerate(THR):
   for li in LEADS:
    j=li-1;v=valid[j]&np.isfinite(p:=pr[ti,j]);
    if not v.any(): continue
    y=(truth[j]>t)&v; dp=(det[j]>t).astype(float);prec,rc,f1,csi,far=_scores(y[v],p[v]);dprec,drc,df1,dcsi,dfar=_scores(y[v],dp[v]);bs=float(np.mean((p[v]-y[v])**2));db=float(np.mean((dp[v]-y[v])**2));cl=float(y[v].mean());bss=1-bs/max(cl*(1-cl),1e-12)
    rec.append({"row_id":r.row_id,"role":r.role,"event":r.independent_system_group,"threshold":float(t),"lead_minutes":li*6,"ensemble_brier":bs,"deterministic_brier":db,"bss_climatology":bss,"ensemble_precision":prec,"ensemble_recall":rc,"ensemble_f1":f1,"ensemble_csi":csi,"ensemble_far":far,"deterministic_f1":df1,"deterministic_csi":dcsi,"fss6":_fss(y,p>=.5,3),"fss18":_fss(y,p>=.5,9),"fss36":_fss(y,p>=.5,18),"det_fss18":_fss(y,dp>=.5,9),"mean_spread":float(sp[j][v].mean()),"mean_abs_error":float(np.abs(em[j][v]-truth[j][v]).mean())})
   # bounded systematic pixel sample for AUC/reliability/CRPS approximation
   for j in range(20):
    v=valid[j].ravel();ind=np.flatnonzero(v)[::64]; yy=(truth[j].ravel()[ind]>t).astype(np.uint8);pp=pr[ti,j].ravel()[ind]
    samples.extend({"row_id":r.row_id,"event":r.independent_system_group,"role":r.role,"threshold":float(t),"lead_minutes":(j+1)*6,"y":int(a),"p":float(b)} for a,b in zip(yy,pp))
  wet_truth=(truth>.1)&valid;wet_det=det>.1; obs=_first_persistent(wet_truth);dtime=_first_persistent(wet_det);etime=_quantile_time_from_marginals(pr[0],.5);lo=_quantile_time_from_marginals(pr[0],.1);hi=_quantile_time_from_marginals(pr[0],.9);active=obs<=120
  if active.any(): onsets.append({"row_id":r.row_id,"event":r.independent_system_group,"role":r.role,"observed_onset_median":float(np.median(obs[active])),"ensemble_onset_median":float(np.median(etime[active])),"deterministic_onset_median":float(np.median(dtime[active])),"ensemble_onset_mae":float(np.mean(np.abs(etime[active]-obs[active]))),"deterministic_onset_mae":float(np.mean(np.abs(dtime[active]-obs[active]))),"ensemble_onset_bias":float(np.mean(etime[active]-obs[active])),"onset_interval_coverage":float(np.mean((obs[active]>=lo[active])&(obs[active]<=hi[active]))),"onset_interval_width":float(np.mean(hi[active]-lo[active])),"p_by30":float(np.mean(pr[0,:5].max(0)[active])),"p_by60":float(np.mean(pr[0,:10].max(0)[active])),"p_by90":float(np.mean(pr[0,:15].max(0)[active])),"p_by120":float(np.mean(pr[0].max(0)[active]))})
  cactive=wet_truth[0]&valid.all(0);cobs=_first_persistent(~wet_truth);cdet=_first_persistent(~wet_det);cmed=_quantile_time_from_marginals(pr[0],.5,wet=False);clo=_quantile_time_from_marginals(pr[0],.1,wet=False);chi=_quantile_time_from_marginals(pr[0],.9,wet=False)
  if cactive.any(): cessations.append({"row_id":r.row_id,"event":r.independent_system_group,"ensemble_cessation_mae":float(np.mean(np.abs(cmed[cactive]-cobs[cactive]))),"deterministic_cessation_mae":float(np.mean(np.abs(cdet[cactive]-cobs[cactive]))),"ensemble_cessation_bias":float(np.mean(cmed[cactive]-cobs[cactive])),"cessation_interval_coverage":float(np.mean((cobs[cactive]>=clo[cactive])&(cobs[cactive]<=chi[cactive]))),"cessation_interval_width":float(np.mean(chi[cactive]-clo[cactive]))})
  if r.role=="hard_negative": hard.append({"row_id":r.row_id,"event":r.independent_system_group,"mean_probability":float(pr[0].mean()),"max_probability":float(pr[0].max()),"wet_area_probability_coverage":float((pr[0]>=.5).mean()),"probabilistic_false_alarm":float((pr[0]>=.5).any()),"deterministic_false_initiation":float((det>.1).any()),"brier":float(np.mean(pr[0]**2))})
  fc=failure[failure.row_id.eq(r.row_id)]
  for x in fc.itertuples(index=False): spreadcat.append({"row_id":r.row_id,"event":r.independent_system_group,"category":x.pysteps_failure_category,"lead_minutes":x.lead_minutes,"spread":float(sp[int(x.lead_minutes/6)-1].mean()),"abs_error":float(np.abs(em[int(x.lead_minutes/6)-1]-truth[int(x.lead_minutes/6)-1]).mean())})
 rd=pd.DataFrame(rec);sa=pd.DataFrame(samples);on=pd.DataFrame(onsets);ce=pd.DataFrame(cessations);hd=pd.DataFrame(hard);sc=pd.DataFrame(spreadcat)
 rd.to_csv(ROOT/"deterministic_and_ensemble_metrics_by_row.csv",index=False);on.to_csv(ROOT/"onset_metrics_by_row.csv",index=False);ce.to_csv(ROOT/"cessation_metrics_by_row.csv",index=False);hd.to_csv(ROOT/"hard_negative_metrics.csv",index=False);sc.to_csv(ROOT/"spread_by_failure_category.csv",index=False)
 ev=rd.groupby(["event","threshold","lead_minutes"],as_index=False).mean(numeric_only=True);ev.to_csv(ROOT/"metrics_by_event.csv",index=False)
 macro=ev.groupby(["threshold","lead_minutes"],as_index=False).mean(numeric_only=True);macro.to_csv(ROOT/"metrics_event_macro.csv",index=False)
 cal=sa.assign(bin=pd.cut(sa.p,np.linspace(0,1,11),include_lowest=True)).groupby(["threshold","lead_minutes","bin"],observed=False).agg(samples=("y","size"),mean_probability=("p","mean"),observed_frequency=("y","mean")).reset_index();cal.to_csv(ROOT/"reliability.csv",index=False)
 auc=sa.groupby(["event","threshold"],as_index=False).apply(lambda x:pd.Series({"auc":_auc(x.y,x.p),"brier":np.mean((x.p-x.y)**2),"sharpness":np.std(x.p)}),include_groups=False);auc.to_csv(ROOT/"probabilistic_metrics_by_event.csv",index=False)
 cat=sc.groupby("category",as_index=False).agg(mean_spread=("spread","mean"),mean_abs_error=("abs_error","mean"),spread_error_correlation=("spread",lambda x:np.nan))
 for i,row in cat.iterrows(): q=sc[sc.category.eq(row.category)];cat.loc[i,"spread_error_correlation"]=q.spread.corr(q.abs_error)
 cat.to_csv(ROOT/"failure_category_uncertainty.csv",index=False)
 # Gaussian approximation supplies a practical continuous-rate CRPS diagnostic from cached mean/spread.
 rr=rd[np.isfinite(rd.mean_spread)&np.isfinite(rd.mean_abs_error)]; z=rr.mean_abs_error/rr.mean_spread.clip(lower=1e-6); crps=rr.mean_spread*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
 crps.to_csv(ROOT/"gaussian_approximate_crps.csv",index=False,header=["crps"])
 summ={"status":"DEVELOPMENT EVALUATION COMPLETE","event_macro_auc":float(auc.groupby("event").auc.mean().mean()),"event_macro_brier":float(auc.groupby("event").brier.mean().mean()),"ensemble_f1":float(ev.groupby("event").ensemble_f1.mean().mean()),"deterministic_f1":float(ev.groupby("event").deterministic_f1.mean().mean()),"ensemble_fss18":float(ev.groupby("event").fss18.mean().mean()),"deterministic_fss18":float(ev.groupby("event").det_fss18.mean().mean()),"ensemble_onset_mae":float(on.groupby("event").ensemble_onset_mae.mean().mean()),"deterministic_onset_mae":float(on.groupby("event").deterministic_onset_mae.mean().mean()),"onset_interval_coverage":float(on.groupby("event").onset_interval_coverage.mean().mean()),"onset_interval_width_minutes":float(on.groupby("event").onset_interval_width.mean().mean()),"ensemble_cessation_mae":float(ce.groupby("event").ensemble_cessation_mae.mean().mean()),"deterministic_cessation_mae":float(ce.groupby("event").deterministic_cessation_mae.mean().mean()),"cessation_interval_coverage":float(ce.groupby("event").cessation_interval_coverage.mean().mean()),"cessation_interval_width_minutes":float(ce.groupby("event").cessation_interval_width.mean().mean()),"gaussian_approximate_crps":float(crps.mean()),"hard_negative_wet_area":float(hd.groupby("event").wet_area_probability_coverage.mean().mean()),"calibration_attempted":False,"calibration_justified":False,"sealed_final_accessed":False}
 summ["classification"]="PROMISING OPERATIONAL BASELINE" if summ["event_macro_brier"]<float(rd.deterministic_brier.mean()) and summ["ensemble_f1"]>=summ["deterministic_f1"]-.02 and summ["hard_negative_wet_area"]<=.01 else "MIXED"
 (ROOT/"stage13_decision.json").write_text(json.dumps(summ,indent=2));outs={p.name:sha256_file(p) for p in ROOT.iterdir() if p.is_file() and p.name!="stage13_manifest.json"};(ROOT/"stage13_manifest.json").write_text(json.dumps({"sealed_final_accessed":False,"outputs":outs},indent=2));print(json.dumps(summ,indent=2))
def main():
 import argparse;p=argparse.ArgumentParser();p.add_argument("phase",choices=["predeclare","materialize","evaluate"]);a=p.parse_args();globals()[a.phase]()
if __name__=="__main__":main()
