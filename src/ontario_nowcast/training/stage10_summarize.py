"""Summarize Stage 10 representation diagnostics without new forecasts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from ..data.manifest import sha256_file

ROOT=Path("artifacts/stage_10");OUT=ROOT/"analysis"
def main():
 m=pd.read_csv(OUT/"representation_metrics_by_row_lead_group.csv");d=pd.read_csv(OUT/"causal_displacement_by_row.csv")
 lead=m.groupby(["role","lead_group","representation"],as_index=False).mean(numeric_only=True);lead.to_csv(OUT/"skill_event_row_macro_by_lead_group.csv",index=False)
 evlead=m.groupby(["role","event_id","lead_group","representation"],as_index=False).mean(numeric_only=True);evlead.to_csv(OUT/"skill_by_event_lead_group.csv",index=False)
 ev=evlead.groupby(["role","event_id","representation"],as_index=False).mean(numeric_only=True);p=ev[ev.role.eq("positive")].pivot(index="event_id",columns="representation",values=["brier","f1","fss_6km","fss_18km","fss_36km","continuous_auc"])
 delta=pd.DataFrame({"event_id":p.index,"brier_improvement":p.brier.raw-p.brier.causal_shift,"f1_improvement":p.f1.causal_shift-p.f1.raw,"fss6_improvement":p.fss_6km.causal_shift-p.fss_6km.raw,"fss18_improvement":p.fss_18km.causal_shift-p.fss_18km.raw,"fss36_improvement":p.fss_36km.causal_shift-p.fss_36km.raw,"auc_improvement":p.continuous_auc.causal_shift-p.continuous_auc.raw});delta.to_csv(OUT/"causal_translation_event_improvements.csv",index=False)
 stage9=pd.read_csv("artifacts/stage_9/source_advantage/oracle_source_advantage_by_row_lead.csv");s=stage9[stage9.collection.eq("stage8_nested_development")];cats=s.groupby("row_id").pysteps_failure_category.agg(lambda x:x.value_counts().index[0]).rename("pysteps_failure_category")
 row=m[m.role.eq("positive")].groupby(["row_id","representation"],as_index=False).mean(numeric_only=True);rp=row.pivot(index="row_id",columns="representation",values=["brier","f1","fss_18km"])
 fr=pd.DataFrame({"pysteps_failure_category":rp.index.map(cats),"brier_improvement":rp.brier.raw-rp.brier.causal_shift,"f1_improvement":rp.f1.causal_shift-rp.f1.raw,"fss18_improvement":rp.fss_18km.causal_shift-rp.fss_18km.raw}).groupby("pysteps_failure_category",as_index=False).agg(rows=("brier_improvement","size"),brier_improvement=("brier_improvement","mean"),f1_improvement=("f1_improvement","mean"),fss18_improvement=("fss18_improvement","mean"));fr.to_csv(OUT/"translation_by_pysteps_failure_category.csv",index=False)
 helpful=s.groupby("row_id").delta_brier_hrrr_vs_pysteps.mean().gt(0).rename("hrrr_helpful");cont=row[row.representation.eq("raw")].set_index("row_id").join(helpful);cont.groupby("hrrr_helpful",as_index=False).agg(rows=("continuous_auc","size"),continuous_auc=("continuous_auc","mean"),continuous_spearman=("continuous_spearman","mean"),binary_f1=("f1","mean"),binary_brier=("brier","mean")).to_csv(OUT/"continuous_apcp_by_source_advantage.csv",index=False)
 hard=m[m.role.eq("hard_negative")].groupby("representation",as_index=False).agg(rows=("row_id","nunique"),brier=("brier","mean"),f1=("f1","mean"),fss6=("fss_6km","mean"),fss18=("fss_18km","mean"),fss36=("fss_36km","mean"));hard=hard.merge(d[d.role.eq("hard_negative")].groupby(lambda _:True).agg(mean_shift_km=("shift_magnitude_km","mean"),meaningful_fraction=("meaningful_match","mean")).reset_index(drop=True),how="cross");hard.to_csv(OUT/"hard_negative_summary.csv",index=False)
 decisions=pd.DataFrame([
  {"representation":"raw continuous hourly APCP","classification":"WEAK / REGIME-DEPENDENT","basis":"Continuous rank skill exists but remains modest and event-dependent; it does not resolve hourly timing or displacement."},
  {"representation":"issue-time displacement-corrected APCP","classification":"NOT USEFUL","basis":"Only a tiny Brier gain; F1, AUC and FSS decline, with inconsistent event effects."},
  {"representation":"higher-temporal-resolution precipitation","classification":"UNAVAILABLE — NOT CLASSIFIED","basis":"Unavailable in the audited hourly forecast archive; PRATE and reflectivity are hourly snapshots, not sub-hourly guidance."},
  {"representation":"hourly radar-like REFC/REFD","classification":"WEAK / REGIME-DEPENDENT","basis":"Complete and causal in the archive but not temporally richer; requires a separately frozen radar-like mapping before skill evaluation."},
 ]);decisions.to_csv(OUT/"representation_decisions.csv",index=False)
 manifest=json.loads((ROOT/"stage10_manifest.json").read_text());manifest["fixed_shift"]["algorithm"]="FFT cross-correlation peak within fixed +/-18-pixel window, scored by occurrence IoU";manifest["outputs"]={p.name:sha256_file(p) for p in list(ROOT.glob("*.csv"))+list(OUT.glob("*.csv"))};manifest["representation_decisions_sha256"]=sha256_file(OUT/"representation_decisions.csv");(ROOT/"stage10_manifest.json").write_text(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
