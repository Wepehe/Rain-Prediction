"""Execute the frozen Stage 8 nested grouped development experiment."""
from __future__ import annotations
import hashlib,json,platform
from pathlib import Path
import numpy as np,pandas as pd,scipy,torch
from scipy.optimize import minimize
from ..data.manifest import sha256_file
from ..evaluation.metrics import categorical_metrics,fractions_skill_score_km
from .stage4b_holdout import _row_metrics

ROOT=Path("artifacts/stage_8"); PREDECL="18e2d0273f5e3a04028339e5cd534d9b2e86a3a8129d32234aaa252e7e94ec5d"; GRID=np.arange(.05,1,.05)
FEATURES=["lead_fraction","current_radar_wet_fraction","a_plus_wet_fraction","a_plus_mean_entropy","pysteps_wet_fraction","hrrr_wet_fraction","a_plus_hrrr_disagreement_fraction","hrrr_radar_forecast_support_fraction"]

def load():
 m=pd.read_csv(ROOT/"development_features/development_feature_manifest.csv"); data={}
 for r in m.itertuples(index=False):
  z=np.load(r.cache_path); data[r.row_id]={k:z[k] for k in z.files}|{"row":r}
 return m,data

def stats(m,data,groups):
 x=np.concatenate([data[r].get("features") for r in m[m.independent_system_group.isin(groups)].row_id]); mean=x.mean(0); std=x.std(0); std[std==0]=1; return mean,std

def quadratic(item):
 a=item["a_plus_probability"].astype(np.float32); h=item["hrrr_occurrence"].astype(np.float32); y=item["target_occurrence"].astype(np.float32); v=item["target_valid"].astype(bool); d=h-a
 A=[];B=[];C=[]
 for j in range(20):
  q=v[j]
  if not np.any(q):
   A.append(np.nan);B.append(np.nan);C.append(np.nan)
   continue
  e=a[j][q]-y[j][q]; dd=d[j][q]; A.append(np.mean(e*e));B.append(np.mean(e*dd));C.append(.25*np.mean(dd*dd))
 return np.array(A),np.array(B),np.array(C)

def fit(m,data,groups):
 mean,std=stats(m,data,groups); subset=m[m.independent_system_group.isin(groups)]; by={g:subset[subset.independent_system_group.eq(g)].row_id.tolist() for g in groups}; q={r:quadratic(data[r]) for r in subset.row_id}
 def fg(beta):
  loss=0.;grad=np.zeros(9)
  for g,ids in by.items():
   gl=0.;gg=np.zeros(9); n=0
   for rid in ids:
    z=(data[rid]["features"]-mean)/std; X=np.c_[np.ones(20),z]; eta=X@beta; gate=1/(1+np.exp(-np.clip(eta,-30,30))); A,B,C=q[rid]
    valid=np.isfinite(A)&np.isfinite(B)&np.isfinite(C)
    if np.any(valid):
     gl+=np.sum(A[valid]+B[valid]*gate[valid]+C[valid]*gate[valid]*gate[valid])
     gg+=X[valid].T@((B[valid]+2*C[valid]*gate[valid])*gate[valid]*(1-gate[valid]));n+=int(valid.sum())
   if n:
    loss+=gl/n;grad+=gg/n
  loss/=len(by);grad/=len(by);loss+=np.sum(beta[1:]**2);grad[1:]+=2*beta[1:]
  return loss,grad
 res=minimize(lambda b:fg(b),np.zeros(9),jac=True,method="L-BFGS-B",options={"maxiter":1000,"gtol":1e-9}); return mean,std,res

def predict(item,mean,std,beta):
 z=(item["features"]-mean)/std; g=1/(1+np.exp(-np.clip(np.c_[np.ones(20),z]@beta,-30,30))); a=item["a_plus_probability"].astype(np.float32);h=item["hrrr_occurrence"].astype(np.float32); return a+.5*g[:,None,None]*(h-a),g

def score_rows(preds,m,data,threshold,model="gate"):
 rows=[]
 for rid,p in preds.items():
  item=data[rid]; r=item["row"]; obs=np.where(item["target_valid"].astype(bool),item["target_occurrence"].astype(float),np.nan); met=_row_metrics(obs,p,threshold); rows.append({"row_id":rid,"role":r.role,"event_id":r.event_id,"independent_system_group":r.independent_system_group,"model":model,"threshold":threshold,**met})
 return pd.DataFrame(rows)

def macro(table):
 pos=table[table.role.eq("positive_initiation")]; metrics=["precision","recall","f1","false_initiation_fraction","onset_mae_minutes","median_onset_bias_minutes","brier_score","detection_within_30min","detection_within_60min","detection_within_90min","detection_within_120min"]
 event=pos.groupby("independent_system_group")[metrics].mean(numeric_only=True); return {k:float(event[k].mean()) for k in metrics},event

def hard_metrics(preds,m,data,threshold,gates=None):
 ids=m[m.role.eq("hard_negative")].row_id; vals=[]
 for rid in ids:
  p=preds[rid]; wet=p>=threshold; vals.append({"row_id":rid,"independent_system_group":data[rid]["row"].independent_system_group,"wet_area_fraction":float(wet.mean()),"false_initiation_fraction":float(wet.any()),"brier_score":float(np.mean(p*p)),"mean_probability":float(p.mean()),"maximum_probability":float(p.max()),"mean_gate_value":float(np.mean(gates[rid])) if gates and rid in gates else np.nan})
 return pd.DataFrame(vals)

def select_threshold(preds,m,data):
 records=[]
 for t in GRID:
  tab=score_rows(preds,m,data,t); ma,_=macro(tab); hard=hard_metrics(preds,m,data,t); records.append({"threshold":t,**ma,"hard_negative_wet_area":float(hard.wet_area_fraction.mean()),"eligible":bool(hard.wet_area_fraction.mean()<=.01 and ma["false_initiation_fraction"]<=.40)})
 c=pd.DataFrame(records); e=c[c.eligible]
 if e.empty: chosen=c.sort_values(["hard_negative_wet_area","false_initiation_fraction","f1"],ascending=[True,True,False]).iloc[0]
 else: chosen=e.sort_values(["f1","onset_mae_minutes","brier_score","precision","threshold"],ascending=[False,True,True,False,True]).iloc[0]
 return float(chosen.threshold),c

def comparator(item,name):
 a=item["a_plus_probability"].astype(np.float32); h=item["hrrr_occurrence"].astype(np.float32); py=(item["pysteps_rate"].astype(np.float32)>=.1).astype(np.float32)
 if name=="A_plus": return a
 if name=="PySTEPS": return py
 if name=="raw_HRRR": return h
 if name=="Stage6_hybrid": return .5*a+.5*h
 conf=np.mean(np.abs(2*a-1),axis=(1,2));g=np.clip(1-conf,0,1);return a+.5*g[:,None,None]*(h-a)

def run():
 spec=ROOT/"gate_predeclaration/stage8_gate_predeclaration.json"; assert sha256_file(spec)==PREDECL
 freeze=json.loads((ROOT/"development_features/development_feature_freeze.json").read_text()); assert freeze["all_causal"] and not freeze["gate_predictions_generated"]
 m,data=load(); groups=sorted(m.independent_system_group.unique()); out=ROOT/"nested_development";out.mkdir(parents=True,exist_ok=True)
 fold_meta=[];outer_preds={};outer_g={};threshold_curves=[]
 for oi,hold in enumerate(groups,1):
  train=[g for g in groups if g!=hold];inner_preds={}
  for ih in train:
   itr=[g for g in train if g!=ih];mean,std,res=fit(m,data,itr)
   for rid in m[m.independent_system_group.eq(ih)].row_id:
    inner_preds[rid]=predict(data[rid],mean,std,res.x)[0]
  inner_m=m[m.independent_system_group.isin(train)];threshold,curve=select_threshold(inner_preds,inner_m,data);curve["outer_held_out_system"]=hold;threshold_curves.append(curve)
  mean,std,res=fit(m,data,train);fold_dir=out/f"fold_{oi:02d}";fold_dir.mkdir(exist_ok=True)
  fold_rows=[]
  for rid in m[m.independent_system_group.eq(hold)].row_id:
   p,g=predict(data[rid],mean,std,res.x);outer_preds[rid]=p;outer_g[rid]=g; path=fold_dir/f"{hashlib.sha256(rid.encode()).hexdigest()[:20]}.npz";np.savez_compressed(path,probability=p.astype(np.float16),prediction=(p>=threshold).astype(np.uint8),gate=g.astype(np.float32),threshold=np.asarray(threshold));fold_rows.append({"row_id":rid,"prediction_path":path.as_posix(),"sha256":sha256_file(path)})
  pd.DataFrame(fold_rows).to_csv(fold_dir/"prediction_manifest.csv",index=False);meta={"outer_system":hold,"training_systems":train,"threshold":threshold,"normalization_mean":dict(zip(FEATURES,mean.tolist())),"normalization_std":dict(zip(FEATURES,std.tolist())),"intercept":float(res.x[0]),"coefficients":dict(zip(FEATURES,res.x[1:].tolist())),"optimizer_success":bool(res.success),"optimizer_status":int(res.status),"optimizer_message":str(res.message),"iterations":int(res.nit),"objective":float(res.fun)};(fold_dir/"fit.json").write_text(json.dumps(meta,indent=2));fold_meta.append(meta);print(f"[stage8-nested] outer {oi}/7 {hold} threshold={threshold:.2f}",flush=True)
 pd.concat(threshold_curves).to_csv(out/"inner_threshold_curves.csv",index=False);(out/"outer_fold_fits.json").write_text(json.dumps(fold_meta,indent=2))
 assert set(outer_preds)==set(m.row_id)
 all_scores=[];threshold_by={x["outer_system"]:x["threshold"] for x in fold_meta}
 for name,t in [("A_plus",.35),("PySTEPS",.5),("raw_HRRR",.5),("Stage6_hybrid",.3)]:
  preds={rid:comparator(data[rid],name) for rid in m.row_id};parts=[score_rows({rid:preds[rid] for rid in m[m.independent_system_group.eq(g)].row_id},m[m.independent_system_group.eq(g)],data,t,name) for g in groups];all_scores.append(pd.concat(parts))
 for name in ["Stage8_gate","uncertainty_rule"]:
  parts=[]
  for g in groups:
   ids=m[m.independent_system_group.eq(g)].row_id;pred={rid:(outer_preds[rid] if name=="Stage8_gate" else comparator(data[rid],"rule")) for rid in ids};parts.append(score_rows(pred,m[m.independent_system_group.eq(g)],data,threshold_by[g],name))
  all_scores.append(pd.concat(parts))
 scores=pd.concat(all_scores,ignore_index=True);scores.to_csv(out/"outer_row_metrics.csv",index=False)
 macros=[];events=[]
 for name,tbl in scores.groupby("model"):
  ma,ev=macro(tbl);macros.append({"model":name,**ma}); ev=ev.reset_index();ev["model"]=name;events.append(ev)
 macro_df=pd.DataFrame(macros);macro_df.to_csv(out/"comparator_event_macro_metrics.csv",index=False);event_df=pd.concat(events);event_df.to_csv(out/"comparator_by_event_metrics.csv",index=False)
 hard=hard_metrics(outer_preds,m,data,0,outer_g);hard["threshold"]=[threshold_by[g] for g in hard.independent_system_group]
 hard["wet_area_fraction"]=[float((outer_preds[r.row_id]>=r.threshold).mean()) for r in hard.itertuples(index=False)]
 hard["false_initiation_fraction"]=[bool((outer_preds[r.row_id]>=r.threshold).any()) for r in hard.itertuples(index=False)]
 hard.to_csv(out/"hard_negative_outer_metrics.csv",index=False)
 gate_scores=scores[scores.model.eq("Stage8_gate")];coh=pd.read_csv(ROOT/"development_features/stage8_development_radar_cohorts.csv");cohort_rows=[]
 for label,ids in [("all_initiation",set(coh.row_id)),("RADAR_LIMITED_INITIATION",set(coh[coh.radar_limited_initiation].row_id)),("RADAR_POOR_INITIATION_V2",set(coh[coh.radar_poor_initiation_v2].row_id))]:
  tab=gate_scores[gate_scores.row_id.isin(ids)]; ma,_=macro(tab) if len(tab) else ({k:np.nan for k in ["precision","recall","f1","false_initiation_fraction","onset_mae_minutes","median_onset_bias_minutes","brier_score","detection_within_30min","detection_within_60min","detection_within_90min","detection_within_120min"]},None);cohort_rows.append({"cohort":label,"rows":len(tab),**ma})
 pd.DataFrame(cohort_rows).to_csv(out/"cohort_metrics.csv",index=False)
 # Gate diagnostics and coefficients
 behavior=[]
 for rid,g in outer_g.items():
  rr=data[rid]["row"];f=data[rid]["features"];a=data[rid]["a_plus_probability"].astype(float);h=data[rid]["hrrr_occurrence"].astype(bool)
  for j,val in enumerate(g): behavior.append({"row_id":rid,"role":rr.role,"event":rr.independent_system_group,"lead_minutes":6*(j+1),"g":val,"a_plus_confidence":float(np.mean(np.abs(2*a[j]-1))),"agreement":float(np.mean((a[j]>=.35)==h[j])),"hrrr_wet_fraction":float(h[j].mean())})
 b=pd.DataFrame(behavior);b.to_csv(out/"gate_behavior.csv",index=False)
 coef=[]
 for fm in fold_meta:
  for name,val in fm["coefficients"].items():coef.append({"outer_system":fm["outer_system"],"feature":name,"coefficient":val})
 pd.DataFrame(coef).to_csv(out/"outer_coefficients.csv",index=False)
 # Broad pixel metrics at fixed leads.
 broad=[]
 for name in ["A_plus","PySTEPS","raw_HRRR","Stage6_hybrid","Stage8_gate","uncertainty_rule"]:
  for lead in (30,60,90,120):
   vals=[]
   for rid in m[m.role.eq("positive_initiation")].row_id:
    item=data[rid];j=lead//6-1;p=outer_preds[rid][j] if name=="Stage8_gate" else comparator(item,"rule" if name=="uncertainty_rule" else name)[j];thr=threshold_by[item["row"].independent_system_group] if name in {"Stage8_gate","uncertainty_rule"} else {"A_plus":.35,"PySTEPS":.5,"raw_HRRR":.5,"Stage6_hybrid":.3}[name];obs=item["target_occurrence"][j].astype(float);pred=(p>=thr).astype(float);met=categorical_metrics(obs,pred,.5);met["fss_18km"]=fractions_skill_score_km(obs,pred,.5,18.0,2.0);vals.append(met)
   broad.append({"model":name,"lead_minutes":lead,"csi":float(np.nanmean([x["csi"] for x in vals])),"f1":float(np.nanmean([x["f1"] for x in vals])),"fss_18km":float(np.nanmean([x["fss_18km"] for x in vals]))})
 pd.DataFrame(broad).to_csv(out/"broad_lead_metrics.csv",index=False)
 mm=macro_df.set_index("model");event_gate=event_df[event_df.model.eq("Stage8_gate")].set_index("independent_system_group");event_a=event_df[event_df.model.eq("A_plus")].set_index("independent_system_group");event_h=event_df[event_df.model.eq("Stage6_hybrid")].set_index("independent_system_group");strict=int((event_gate.f1-event_a.f1>0).sum());gate_f=float(mm.loc["Stage8_gate","f1"]);a_f=float(mm.loc["A_plus","f1"]);hy_f=float(mm.loc["Stage6_hybrid","f1"]);false=float(mm.loc["Stage8_gate","false_initiation_fraction"]);hw=float(hard.wet_area_fraction.mean());conditions={"A_vs_Aplus":gate_f>=a_f+.01,"B_vs_hybrid":gate_f>=hy_f+.005,"C_four_strictly_positive":strict>=4,"D_hard_negative_wet_area":hw<=.01,"E_false_initiation":false<=.40};promote=all(conditions.values());classification="PROMOTED — CONDITIONAL HRRR GATE" if promote else ("NEGATIVE — NOT PROMOTED" if gate_f<=a_f and gate_f<=hy_f else ("NULL — NOT PROMOTED" if abs(gate_f-a_f)<.01 and abs(gate_f-hy_f)<.005 else "MIXED — NOT PROMOTED"))
 decision={"classification":classification,"promoted":promote,"conditions":conditions,"values":{"stage8_event_macro_f1":gate_f,"a_plus_event_macro_f1":a_f,"hybrid_event_macro_f1":hy_f,"strictly_positive_systems":strict,"hard_negative_wet_area":hw,"false_initiation_fraction":false},"final_set_scored":False,"predeclaration_sha256":PREDECL};(out/"promotion_decision.json").write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))

if __name__=="__main__":run()
