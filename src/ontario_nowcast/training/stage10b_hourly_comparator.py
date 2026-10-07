"""Evaluate frozen hourly binary APCP context on Stage 10B native targets."""
from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
from ..evaluation.metrics import categorical_metrics
OFF=np.arange(15,121,15);OUT=Path("artifacts/stage_10b/evaluation")
def interp(frames,minutes):
 p=minutes/6-1;lo=max(0,int(np.floor(p)));hi=min(len(frames)-1,int(np.ceil(p)));w=p-lo;return frames[lo]*(1-w)+frames[hi]*w
def met(obs,fc):
 q=np.isfinite(obs);m=categorical_metrics(obs[q],fc[q],.1);o=obs[q]>=.1;f=fc[q]>=.1
 return {k:m[k] for k in ["csi","pod","far","f1"]}|{"brier":float(np.mean((f.astype(float)-o.astype(float))**2)),"forecast_wet_fraction":float(f.mean()),"observed_wet_fraction":float(o.mean())}
def run():
 man=pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv");rows=[]
 for r in man.itertuples(index=False):
  z=np.load(r.cache_path);target=z["target_rate"].astype(float);h=z["hrrr_occurrence"].astype(float);cum=np.concatenate([np.zeros_like(target[:1]),np.cumsum(np.nan_to_num(target)*.1,axis=0)])
  for minute in OFF:
   j=min(19,int(np.ceil(minute/6))-1);fc=h[j];rate=interp(target,minute);ce=interp(cum,minute+6);cs=np.zeros_like(ce) if minute==15 else interp(cum,minute-15+6);interval=(ce-cs)*4
   for truth,obs in [("endpoint_rate",rate),("interval_accumulation",interval)]:rows.append({"row_id":r.row_id,"role":"positive" if r.role=="positive_initiation" else "hard_negative","event_id":r.independent_system_group,"offset_minutes":minute,"lead_group":f"{(minute-1)//30*30}-{((minute-1)//30+1)*30}","truth_semantics":truth,**met(obs,fc)})
 out=pd.DataFrame(rows);out.to_csv(OUT/"hourly_apcp_native_target_comparator.csv",index=False)
if __name__=="__main__":run()
