"""Summarize Stage 10B against the frozen hourly APCP diagnosis."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from ..data.manifest import sha256_file
ROOT=Path("artifacts/stage_10b");OUT=ROOT/"evaluation"
def main():
 sub=pd.read_csv(OUT/"native_metrics_by_row_offset.csv");hour=pd.read_csv(OUT/"hourly_apcp_native_target_comparator.csv");srow=sub.groupby(["row_id","role","event_id","representation","lead_group"],as_index=False).mean(numeric_only=True)
 hour_parts=[]
 for rep,truth in [("subhourly_APCP","interval_accumulation"),("subhourly_PRATE","endpoint_rate"),("subhourly_REFC","endpoint_rate")]:
  h=hour[hour.truth_semantics.eq(truth)].groupby(["row_id","role","event_id","lead_group"],as_index=False).mean(numeric_only=True);h["representation"]=rep;hour_parts.append(h)
 hrow=pd.concat(hour_parts).rename(columns={"brier":"hourly_brier","f1":"hourly_f1","csi":"hourly_csi","pod":"hourly_pod","far":"hourly_far"})
 joined=srow.merge(hrow[["row_id","role","event_id","representation","lead_group","hourly_brier","hourly_f1","hourly_csi","hourly_pod","hourly_far"]],on=["row_id","role","event_id","representation","lead_group"],how="left");joined["f1_minus_hourly"]=joined.f1-joined.hourly_f1;joined["brier_improvement_vs_hourly"]=joined.hourly_brier-joined.brier;joined.to_csv(OUT/"subhourly_vs_hourly_by_row_group.csv",index=False)
 ev=joined[joined.role.eq("positive")].groupby(["event_id","representation"],as_index=False).mean(numeric_only=True);ev.to_csv(OUT/"subhourly_vs_hourly_by_event.csv",index=False);ev.groupby("representation",as_index=False).agg(events=("event_id","nunique"),mean_f1=("f1","mean"),mean_hourly_f1=("hourly_f1","mean"),mean_f1_change=("f1_minus_hourly","mean"),systems_f1_improved=("f1_minus_hourly",lambda x:int((x>0).sum())),mean_brier_improvement=("brier_improvement_vs_hourly","mean")).to_csv(OUT/"subhourly_vs_hourly_event_consistency.csv",index=False)
 stage9=pd.read_csv("artifacts/stage_9/source_advantage/oracle_source_advantage_by_row_lead.csv");cats=stage9[stage9.collection.eq("stage8_nested_development")].groupby("row_id").pysteps_failure_category.agg(lambda x:x.value_counts().index[0]).rename("pysteps_failure_category");jc=joined[joined.role.eq("positive")].join(cats,on="row_id");jc.groupby(["pysteps_failure_category","representation"],as_index=False).agg(rows=("row_id","nunique"),f1_change_vs_hourly=("f1_minus_hourly","mean"),brier_improvement_vs_hourly=("brier_improvement_vs_hourly","mean")).to_csv(OUT/"subhourly_by_pysteps_failure_category.csv",index=False)
 onset=pd.read_csv(OUT/"onset_by_row.csv");hard=joined[joined.role.eq("hard_negative")].groupby("representation",as_index=False).agg(rows=("row_id","nunique"),false_wet_area=("forecast_wet_fraction","mean"),brier=("brier","mean"));ho=onset[onset.role.eq("hard_negative")].groupby("representation",as_index=False).agg(false_initiation=("false_initiation_fraction","mean"),forecast_detection_coverage=("detected_fraction","mean"));hard.merge(ho,on="representation").to_csv(OUT/"hard_negative_summary.csv",index=False)
 consistency=pd.read_csv(OUT/"subhourly_vs_hourly_event_consistency.csv").set_index("representation");dec=[]
 for name in ["subhourly_APCP","subhourly_PRATE","subhourly_REFC"]:
  r=consistency.loc[name];classification="PROMISING" if r.mean_f1_change>0 and r.systems_f1_improved>=4 else ("WEAK / REGIME-DEPENDENT" if r.systems_f1_improved>=2 else "NOT USEFUL");dec.append({"representation":name,"classification":classification,"mean_f1_change_vs_hourly":r.mean_f1_change,"systems_f1_improved":int(r.systems_f1_improved),"mean_brier_improvement_vs_hourly":r.mean_brier_improvement})
 pd.DataFrame(dec).to_csv(OUT/"representation_decisions.csv",index=False)
 manifest=json.loads((ROOT/"stage10b_evaluation_manifest.json").read_text());manifest["source_coverage"]=json.loads((ROOT/"wrfsubhf_audit_summary.json").read_text());manifest["smoke_gate"]=json.loads((ROOT/"representative_smoke_gate.json").read_text());manifest["outputs"]={p.name:sha256_file(p) for p in OUT.glob("*.csv")};manifest["decisions_sha256"]=sha256_file(OUT/"representation_decisions.csv");(ROOT/"stage10b_final_manifest.json").write_text(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
