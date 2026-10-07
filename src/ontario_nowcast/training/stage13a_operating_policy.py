"""Stage 13A: nested operating policy and exact frozen-member timing."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.ndimage import uniform_filter
from ..data.manifest import sha256_file
from ..models.baselines import pysteps_steps_ensemble
from ..sample_evaluation import _downsample_max
from .stage13_probabilistic_pysteps import _event_path,_seed,rows

ROOT=Path("artifacts/stage_13a"); S13=Path("artifacts/stage_13"); CUTS=[.125,.25,.375,.5,.625,.75,.875]; EXPECTED="d78f57bb212f5ad2cead122920b2e0cd2f9f9397f79169cd3dbfc6796f5bcb2c"
def first(a):
 h=a[..., :-1,:,:]&a[...,1:,:,:] if a.ndim==4 else a[:-1]&a[1:];o=np.full(a.shape[-2:],126,dtype=np.int16) if a.ndim==3 else np.full((a.shape[0],*a.shape[-2:]),126,dtype=np.int16);ok=h.any(-3);o[ok]=(h.argmax(-3)[ok]+1)*6;return o
def score(y,q):
 tp=np.sum(y&q);fp=np.sum(~y&q);fn=np.sum(y&~q);p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);return {"precision":p,"pod":r,"far":fp/max(tp+fp,1),"f1":2*p*r/max(p+r,1e-12),"csi":tp/max(tp+fp+fn,1)}
def fss(y,q,n):
 size=(1,n,n);a=uniform_filter(y.astype(np.float32),size,mode="constant");b=uniform_filter(q.astype(np.float32),size,mode="constant");return 1-np.mean((a-b)**2)/max(np.mean(a*a+b*b),1e-12)
def policy_table():
 man=pd.read_csv(S13/"ensemble_manifest.csv");bm=pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv").set_index("row_id");out=[]
 for r in man.itertuples(index=False):
  e=np.load(r.cache_path);z=np.load(bm.loc[r.row_id,"cache_path"]);p=e["probability"][0].astype(float);d=z["pysteps_rate"].astype(float)>=.1;y=z["target_rate"].astype(float)>=.1;v=z["target_valid"].astype(bool);ot=first(y&v);dt=first(d)
  for name,cut in [("deterministic",np.nan),("fixed_0.5",.5)]+[(f"cut_{c}",c) for c in CUTS]:
   q=d if name=="deterministic" else p>=cut;s=score(y[v],q[v]);s.update({"row_id":r.row_id,"role":r.role,"event":r.independent_system_group,"policy":name,"threshold":cut,"fss6":fss(y,q,3) if name in ["deterministic","fixed_0.5"] else np.nan,"fss18":fss(y,q,9),"fss36":fss(y,q,18) if name in ["deterministic","fixed_0.5"] else np.nan})
   qt=first(q);active=ot<=120;s["onset_mae"]=float(np.mean(np.abs(qt[active]-ot[active]))) if active.any() else np.nan;s["onset_bias"]=float(np.mean(qt[active]-ot[active])) if active.any() else np.nan;s["wet_area"]=float(q.mean());s["false_initiation"]=float(q.any()) if r.role=="hard_negative" else np.nan;out.append(s)
 return pd.DataFrame(out)
def select_nested(t):
 groups=sorted(t.event.unique());sel=[]
 for outer in groups:
  train=t[t.event.ne(outer)];candidates=[]
  for c in CUTS:
   q=train[train.policy.eq(f"cut_{c}")];pos=q[q.role.eq("positive")];neg=q[q.role.eq("hard_negative")];ev=pos.groupby("event").mean(numeric_only=True)
   candidates.append({"threshold":c,"f1":ev.f1.mean(),"fss18":ev.fss18.mean(),"far":ev.far.mean(),"onset_mae":ev.onset_mae.mean(),"negative_wet_area":neg.wet_area.mean() if len(neg) else 0.,"negative_false_initiation":neg.false_initiation.mean() if len(neg) else 0.})
  x=pd.DataFrame(candidates);eligible=x[(x.negative_wet_area<=.01)&(x.negative_false_initiation<=.40)];
  if eligible.empty: raise RuntimeError(f"no eligible threshold for {outer}")
  best=eligible.sort_values(["f1","fss18","far","onset_mae","threshold"],ascending=[False,False,True,True,False]).iloc[0];sel.append({"outer_system":outer,**best.to_dict(),"training_systems":"|".join(g for g in groups if g!=outer)})
 return pd.DataFrame(sel)
def add_outer_spatial(chosen):
 bm=pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv").set_index("row_id");paths=pd.read_csv(S13/"ensemble_manifest.csv").set_index("row_id")
 for i,r in chosen.iterrows():
  p=np.load(paths.loc[r.row_id,"cache_path"])["probability"][0].astype(float);z=np.load(bm.loc[r.row_id,"cache_path"]);y=z["target_rate"].astype(float)>=.1;q=p>=r.outer_threshold;chosen.loc[i,"fss6"]=fss(y,q,3);chosen.loc[i,"fss36"]=fss(y,q,18)
 return chosen
def trajectories():
 man=pd.read_csv(S13/"ensemble_manifest.csv");bm=pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv").set_index("row_id");rr=rows().set_index("row_id");ev={};tim=[];blind=[];checks=[]
 fail=pd.read_csv("artifacts/stage_9/source_advantage/oracle_source_advantage_by_row_lead.csv");fail=fail[fail.collection.eq("stage8_nested_development")]
 for i,r in enumerate(man.itertuples(index=False),1):
  meta=rr.loc[r.row_id]
  if meta.event_id not in ev:
   z=np.load(_event_path(meta.event_id,meta.radar_source_path));ev[meta.event_id]=(_downsample_max(z["rate_mm_hr"],2),pd.to_datetime(z["times"],utc=True))
  rate,times=ev[meta.event_id];ix=times.get_indexer([pd.Timestamp(meta.issue_time_utc)])[0];y0,y1,x0,x1=map(int,[meta.tile_y_min,meta.tile_y_max,meta.tile_x_min,meta.tile_x_max]);ens=pysteps_steps_ensemble(rate[ix-9:ix+1,y0:y1+1,x0:x1+1],20,km_per_pixel=2,timestep_minutes=6,ensemble_members=8,rain_threshold=.1,seed=_seed(r.row_id));wet=ens>.1
  ce=np.load(r.cache_path);cp=ce["probability"].astype(float);cm=ce["ensemble_mean"].astype(float);cs=ce["ensemble_spread"].astype(float);rp=np.stack([(ens>x).mean(0) for x in [.1,1,2.5,5]],0);finite=np.isfinite(ens)
  if i<=5: checks.append({"row_id":r.row_id,"probability_max_abs":float(np.nanmax(np.abs(rp-cp))),"mean_max_abs":float(np.nanmax(np.abs(ens.mean(0)-cm))),"spread_max_abs":float(np.nanmax(np.abs(ens.std(0)-cs))),"nan_mask_consistent":bool(np.array_equal(np.isnan(ens.mean(0)),np.isnan(cm)))})
  z=np.load(bm.loc[r.row_id,"cache_path"]);truth=z["target_rate"].astype(float);valid=z["target_valid"].astype(bool);obs=first((truth>.1)&valid);mt=first(wet);active=obs<=120
  if active.any():
   qlo=np.quantile(mt,.125,axis=0);med=np.median(mt,axis=0);qhi=np.quantile(mt,.875,axis=0);vals=mt[:,active]
   tim.append({"row_id":r.row_id,"event":r.independent_system_group,"role":r.role,"kind":"onset","median_mae":float(np.mean(np.abs(med[active]-obs[active]))),"median_bias":float(np.mean(med[active]-obs[active])),"interval_coverage":float(np.mean((obs[active]>=qlo[active])&(obs[active]<=qhi[active]))),"interval_width":float(np.mean(qhi[active]-qlo[active])),"no_member_fraction":float(np.mean((vals>120).all(0))),"by30":float(np.mean(vals<=30)),"by60":float(np.mean(vals<=60)),"by90":float(np.mean(vals<=90)),"by120":float(np.mean(vals<=120)),"earliest":float(np.mean(vals.min(0))),"latest":float(np.mean(vals.max(0))),"q12_5":float(np.mean(qlo[active])),"median":float(np.mean(med[active])),"q87_5":float(np.mean(qhi[active]))})
  cactive=((truth[0]>.1)&valid.all(0));co=first(~(truth>.1));ct=first(~wet)
  if cactive.any():
   qlo=np.quantile(ct,.125,axis=0);med=np.median(ct,axis=0);qhi=np.quantile(ct,.875,axis=0);tim.append({"row_id":r.row_id,"event":r.independent_system_group,"role":r.role,"kind":"cessation","median_mae":float(np.mean(np.abs(med[cactive]-co[cactive]))),"median_bias":float(np.mean(med[cactive]-co[cactive])),"interval_coverage":float(np.mean((co[cactive]>=qlo[cactive])&(co[cactive]<=qhi[cactive]))),"interval_width":float(np.mean(qhi[cactive]-qlo[cactive]))})
  nf=fail[(fail.row_id.eq(r.row_id))&fail.pysteps_failure_category.eq("no_initiation_signal")]
  if len(nf):
   for lead in nf.lead_minutes.unique():
    j=int(lead/6)-1;counts=wet[:,j].sum(0);blind.append({"row_id":r.row_id,"event":r.independent_system_group,"lead_minutes":lead,"zero_members_wet":float(np.mean(counts==0)),"at_least_one_wet":float(np.mean(counts>=1)),"at_least_four_wet":float(np.mean(counts>=4))})
  if i%10==0:print(f"[stage13a] {i}/{len(man)}",flush=True)
 pd.DataFrame(tim).to_csv(ROOT/"exact_trajectory_timing.csv",index=False);pd.DataFrame(blind).to_csv(ROOT/"radar_blind_member_behavior.csv",index=False);pd.DataFrame(checks).to_csv(ROOT/"regeneration_checks.csv",index=False)
def verify_only(n=5):
 man=pd.read_csv(S13/"ensemble_manifest.csv").head(n);rr=rows().set_index("row_id");ev={};checks=[]
 for r in man.itertuples(index=False):
  m=rr.loc[r.row_id]
  if m.event_id not in ev:
   z=np.load(_event_path(m.event_id,m.radar_source_path));ev[m.event_id]=(_downsample_max(z["rate_mm_hr"],2),pd.to_datetime(z["times"],utc=True))
  rate,times=ev[m.event_id];ix=times.get_indexer([pd.Timestamp(m.issue_time_utc)])[0];y0,y1,x0,x1=map(int,[m.tile_y_min,m.tile_y_max,m.tile_x_min,m.tile_x_max]);ens=pysteps_steps_ensemble(rate[ix-9:ix+1,y0:y1+1,x0:x1+1],20,km_per_pixel=2,timestep_minutes=6,ensemble_members=8,rain_threshold=.1,seed=_seed(r.row_id));c=np.load(r.cache_path);rp=np.stack([(ens>x).mean(0) for x in [.1,1,2.5,5]],0);mean=ens.mean(0);spread=ens.std(0)
  checks.append({"row_id":r.row_id,"probability_max_abs":float(np.nanmax(np.abs(rp-c["probability"].astype(float)))),"mean_max_abs":float(np.nanmax(np.abs(mean-c["ensemble_mean"].astype(float)))),"spread_max_abs":float(np.nanmax(np.abs(spread-c["ensemble_spread"].astype(float)))),"mean_nan_mask_consistent":bool(np.array_equal(np.isnan(mean),np.isnan(c["ensemble_mean"]))),"spread_nan_mask_consistent":bool(np.array_equal(np.isnan(spread),np.isnan(c["ensemble_spread"])))})
 pd.DataFrame(checks).to_csv(ROOT/"regeneration_checks.csv",index=False)
def run():
 ROOT.mkdir(parents=True,exist_ok=True)
 if sha256_file(S13/"stage13_predeclaration.json")!=EXPECTED:raise RuntimeError("Stage 13 predeclaration hash changed")
 old=json.loads((S13/"stage13_decision.json").read_text());assert old["classification"]=="MIXED" and not old["sealed_final_accessed"]
 t=policy_table();t.to_csv(ROOT/"all_threshold_metrics_by_row.csv",index=False);sel=select_nested(t);sel.to_csv(ROOT/"outer_threshold_selection.csv",index=False)
 chosen=pd.concat([t[(t.event.eq(x.outer_system))&(t.policy.eq(f"cut_{x.threshold}"))].assign(outer_threshold=x.threshold) for x in sel.itertuples(index=False)]);chosen=add_outer_spatial(chosen);chosen.to_csv(ROOT/"outer_policy_metrics_by_row.csv",index=False)
 trajectories(); pos=chosen[chosen.role.eq("positive")];ev=pos.groupby("event",as_index=False).mean(numeric_only=True);det=t[(t.role.eq("positive"))&t.policy.eq("deterministic")].groupby("event",as_index=False).mean(numeric_only=True);fix=t[(t.role.eq("positive"))&t.policy.eq("fixed_0.5")].groupby("event",as_index=False).mean(numeric_only=True);cmp=det.merge(fix,on="event",suffixes=("_det","_fixed")).merge(ev,on="event");cmp["f1_delta_vs_det"]=cmp.f1-cmp.f1_det;cmp["fss18_delta_vs_det"]=cmp.fss18-cmp.fss18_det;cmp.to_csv(ROOT/"comparison_by_event.csv",index=False)
 neg=chosen[chosen.role.eq("hard_negative")];tim=pd.read_csv(ROOT/"exact_trajectory_timing.csv");on=tim[tim.kind.eq("onset")].groupby("event").mean(numeric_only=True).mean();ce=tim[tim.kind.eq("cessation")].groupby("event").mean(numeric_only=True).mean();brier=pd.read_csv(S13/"metrics_by_event.csv");brierwins=int((brier.groupby("event").deterministic_brier.mean()>brier.groupby("event").ensemble_brier.mean()).sum())
 gates={"A_f1":bool(cmp.f1.mean()>=cmp.f1_det.mean()-.005),"B_fss18":bool(cmp.fss18.mean()>=cmp.fss18_det.mean()-.01),"C_event_consistency":bool((cmp.f1_delta_vs_det>=-.005).sum()>=4),"D_brier_systems":bool(brierwins>=6),"E_hard_negative_wet_area":bool(neg.wet_area.mean()<=.01),"F_hard_negative_false_initiation":bool(neg.false_initiation.mean()<=.40)}
 timing_class="USEFUL" if on.interval_coverage>=.7 and on.median_mae<=old["ensemble_onset_mae"] else "UNDER-DISPERSED" if on.interval_coverage<.7 else "UNINFORMATIVE";decision={"status":"STAGE 13A COMPLETE","classification":"PROMISING OPERATIONAL BASELINE" if all(gates.values()) else "MIXED — CLOSE" if sum(gates.values())>=4 else "NOT USEFUL","timing_classification":timing_class,"gates":gates,"systems_improved":int((cmp.f1_delta_vs_det>0).sum()),"systems_tied_within_0.005":int((cmp.f1_delta_vs_det.abs()<=.005).sum()),"systems_worsened":int((cmp.f1_delta_vs_det<0).sum()),"nested_f1":float(cmp.f1.mean()),"deterministic_f1":float(cmp.f1_det.mean()),"nested_fss18":float(cmp.fss18.mean()),"deterministic_fss18":float(cmp.fss18_det.mean()),"exact_onset":on.to_dict(),"exact_cessation":ce.to_dict(),"hard_negative_wet_area":float(neg.wet_area.mean()),"hard_negative_false_initiation":float(neg.false_initiation.mean()),"brier_systems_improved":brierwins,"merits_final_evaluation":bool(all(gates.values())),"sealed_final_accessed":False}
 (ROOT/"stage13a_decision.json").write_text(json.dumps(decision,indent=2));outs={p.name:sha256_file(p) for p in ROOT.iterdir() if p.is_file() and p.name!="stage13a_manifest.json"};(ROOT/"stage13a_manifest.json").write_text(json.dumps({"stage13_predeclaration_sha256":EXPECTED,"sealed_final_accessed":False,"outputs":outs},indent=2));print(json.dumps(decision,indent=2))
def finalize():
 d=json.loads((ROOT/"stage13a_decision.json").read_text());d["exact_cessation"]={k:v for k,v in d["exact_cessation"].items() if k in ["median_mae","median_bias","interval_coverage","interval_width"]};(ROOT/"stage13a_decision.json").write_text(json.dumps(d,indent=2,allow_nan=False))
 t=pd.read_csv(ROOT/"all_threshold_metrics_by_row.csv");c=[]
 for cut in CUTS:
  q=t[t.policy.eq(f"cut_{cut}")];pos=q[q.role.eq("positive")].groupby("event").mean(numeric_only=True);neg=q[q.role.eq("hard_negative")]
  c.append({"threshold":cut,"f1":pos.f1.mean(),"fss18":pos.fss18.mean(),"far":pos.far.mean(),"onset_mae":pos.onset_mae.mean(),"negative_wet_area":neg.wet_area.mean(),"negative_false_initiation":neg.false_initiation.mean()})
 x=pd.DataFrame(c);e=x[(x.negative_wet_area<=.01)&(x.negative_false_initiation<=.40)].sort_values(["f1","fss18","far","onset_mae","threshold"],ascending=[False,False,True,True,False]);deploy=float(e.iloc[0].threshold);x.to_csv(ROOT/"deployment_threshold_selection.csv",index=False)
 s13=json.loads((S13/"stage13_decision.json").read_text());checks=pd.read_csv(ROOT/"regeneration_checks.csv");audit={"stage13_predeclaration_sha256":sha256_file(S13/"stage13_predeclaration.json"),"predeclaration_matches":sha256_file(S13/"stage13_predeclaration.json")==EXPECTED,"stage13_decision_sha256":sha256_file(S13/"stage13_decision.json"),"reproduced_stage13_metrics":{"deterministic_f1":s13["deterministic_f1"],"raw_ensemble_brier":s13["event_macro_brier"],"fixed_0_5_f1":s13["ensemble_f1"],"deterministic_fss18":s13["deterministic_fss18"],"fixed_0_5_fss18":s13["ensemble_fss18"],"hard_negative_wet_area":s13["hard_negative_wet_area"]},"regenerated_sample_rows":len(checks),"probabilities_exact":bool((checks.probability_max_abs==0).all()),"mean_nan_masks_exact":bool(checks.mean_nan_mask_consistent.all()),"spread_nan_masks_exact":bool(checks.spread_nan_mask_consistent.all()),"mean_matches_float16_quantization":bool((checks.mean_max_abs<=.03125).all()),"spread_matches_float16_quantization":bool((checks.spread_max_abs<=.015625).all()),"sealed_final_accessed":False};(ROOT/"input_reproduction_audit.json").write_text(json.dumps(audit,indent=2))
 proc={"status":"FROZEN_AFTER_STAGE13A_DEVELOPMENT_PROMOTION","ensemble_configuration":"unchanged Stage 13 predeclaration","stage13_predeclaration_sha256":EXPECTED,"seed_policy":"uint32(first 8 hex SHA256(row_id + ':stage13:v1'))","deployment_probability_threshold":deploy,"deployment_threshold_source":"all-development grouped event-macro selection over frozen 0.125..0.875 grid after nested promotion","probability_products_mm_hr":[.1,1.,2.5,5.],"categorical_policy":f"P(rate > 0.1 mm/h) >= {deploy}","persistence":"two consecutive six-minute frames","exact_timing":"empirical first persistent wet / cessation time for each of eight members; 12.5/50/87.5 percentiles","timing_classification":"UNDER-DISPERSED","evaluation_code":"src/ontario_nowcast/training/stage13a_operating_policy.py","evaluation_code_sha256":sha256_file(Path(__file__)),"ensemble_manifest_sha256":sha256_file(S13/"ensemble_manifest.csv"),"outer_threshold_selection_sha256":sha256_file(ROOT/"outer_threshold_selection.csv"),"stage13a_decision_sha256":sha256_file(ROOT/"stage13a_decision.json"),"sealed_final_accessed":False,"final_evaluation_authorized_in_this_pass":False};(ROOT/"final_procedure_manifest.json").write_text(json.dumps(proc,indent=2))
 outs={p.name:sha256_file(p) for p in ROOT.iterdir() if p.is_file() and p.name!="stage13a_manifest.json"};(ROOT/"stage13a_manifest.json").write_text(json.dumps({"stage13_predeclaration_sha256":EXPECTED,"sealed_final_accessed":False,"outputs":outs},indent=2));print(json.dumps({"classification":d["classification"],"deployment_threshold":deploy,"final_procedure_sha256":sha256_file(ROOT/"final_procedure_manifest.json")},indent=2))
if __name__=="__main__":
 import sys
 verify_only() if len(sys.argv)>1 and sys.argv[1]=="verify" else finalize() if len(sys.argv)>1 and sys.argv[1]=="finalize" else run()
