"""Complete Stage 9 post-mortem summaries from diagnostic caches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..data.manifest import sha256_file
from .stage8_nested_run import FEATURES as STAGE8_FEATURES


ROOT=Path("artifacts/stage_9/source_advantage")


def main():
    oracle=pd.read_csv(ROOT/"oracle_source_advantage_by_row_lead.csv")
    pos=oracle[oracle.role.ne("hard_negative")].copy()
    failures=pos[pos.pysteps_failure_category.ne("other_or_adequate")].copy()
    d=failures.delta_brier_hrrr_vs_pysteps
    failures["hrrr_response"] = np.select([d>.01,(d>0)&(d<=.01),d<-.01],["corrects","partially_corrects","worsens"],default="also_misses")
    response=failures.groupby(["collection","pysteps_failure_category","hrrr_response"]).size().unstack(fill_value=0)
    response=response.div(response.sum(axis=1),axis=0).reset_index();response.to_csv(ROOT/"pysteps_failure_hrrr_response_fractions.csv",index=False)
    # Make the coarse first-hour timing mechanism explicit in the descriptive HRRR taxonomy.
    timing=(pos.hrrr_failure_category.eq("false_precipitation") & pos.lead_minutes.le(60))
    pos.loc[timing,"hrrr_failure_category"]="hourly_signal_too_early"
    pos.groupby(["collection","hrrr_failure_category"],as_index=False).agg(count=("row_id","size"),events=("event_id","nunique"),hrrr_win_fraction=("hrrr_beats_pysteps","mean"),mean_delta_brier=("delta_brier_hrrr_vs_pysteps","mean")).to_csv(ROOT/"hrrr_failure_analysis_detailed.csv",index=False)

    # Exact frozen Stage 8 feature post-mortem.
    m=pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv"); records=[]
    oracle8=oracle[oracle.collection.eq("stage8_nested_development")].set_index(["row_id","lead_minutes"])
    for r in m.itertuples(index=False):
        z=np.load(r.cache_path); f=z["features"].astype(float)
        for j in range(20):
            key=(r.row_id,6*(j+1))
            if key not in oracle8.index: continue
            target=float(oracle8.loc[key,"delta_brier_hrrr_vs_pysteps"])
            records.append({"row_id":r.row_id,"event_id":r.independent_system_group,"lead_minutes":6*(j+1),"oracle_advantage":target,**dict(zip(STAGE8_FEATURES,f[j]))})
    sf=pd.DataFrame(records);sf.to_csv(ROOT/"stage8_features_with_oracle_advantage.csv",index=False)
    assoc=[];var=[]
    for name in STAGE8_FEATURES:
        rho=spearmanr(sf[name],sf.oracle_advantage,nan_policy="omit").statistic; signs=[]
        for _,t in sf.groupby("event_id"): signs.append(spearmanr(t[name],t.oracle_advantage,nan_policy="omit").statistic)
        em=sf.groupby("event_id")[name].mean();between=em.var();within=sf.groupby("event_id")[name].var().mean()
        assoc.append({"feature":name,"spearman_all":rho,"positive_event_associations":int(np.sum(np.asarray(signs)>0)),"negative_event_associations":int(np.sum(np.asarray(signs)<0))})
        var.append({"feature":name,"mean":sf[name].mean(),"std":sf[name].std(),"min":sf[name].min(),"max":sf[name].max(),"between_event_variance":between,"within_event_variance":within,"between_fraction":between/(between+within) if between+within else np.nan})
    pd.DataFrame(assoc).to_csv(ROOT/"stage8_exact_feature_oracle_associations.csv",index=False);pd.DataFrame(var).to_csv(ROOT/"stage8_exact_feature_distributions_variance.csv",index=False)

    # Frozen Stage 8 objective versus promotion endpoint.
    ev=pd.read_csv("artifacts/stage_8/nested_development/comparator_by_event_metrics.csv")
    piv=ev.pivot(index="independent_system_group",columns="model",values=["f1","recall","onset_mae_minutes","brier_score"])
    align=pd.DataFrame({"event_id":piv.index,"brier_improvement":piv["brier_score"]["A_plus"]-piv["brier_score"]["Stage8_gate"],"f1_improvement":piv["f1"]["Stage8_gate"]-piv["f1"]["A_plus"],"recall_improvement":piv["recall"]["Stage8_gate"]-piv["recall"]["A_plus"],"onset_mae_improvement":piv["onset_mae_minutes"]["A_plus"]-piv["onset_mae_minutes"]["Stage8_gate"]})
    align["brier_and_f1_agree"]=np.sign(align.brier_improvement)==np.sign(align.f1_improvement);align.to_csv(ROOT/"stage8_objective_endpoint_alignment.csv",index=False)

    fits=json.loads(Path("artifacts/stage_8/nested_development/outer_fold_fits.json").read_text());reg=[]
    for f in fits:
        coeff=np.asarray(list(f["coefficients"].values())); penalty=float(np.sum(coeff*coeff));reg.append({"outer_system":f["outer_system"],"objective":f["objective"],"lambda_penalty":penalty,"penalty_fraction_of_objective":penalty/f["objective"],"max_abs_standardized_coefficient":float(np.max(np.abs(coeff)))})
    pd.DataFrame(reg).to_csv(ROOT/"stage8_regularization_diagnostics.csv",index=False)

    macro=pd.read_csv(ROOT/"advantage_predictor_event_macro.csv").iloc[0]
    conclusion={"classification":"WEAK / REGIME-DEPENDENT","reason":"OOF event-macro AUC is weak and variable; the classifier makes the same source choice as always-PySTEPS and fails to beat the fixed hybrid on Brier.","event_macro_auc":macro.mean_auc,"chosen_forecast_brier":macro.mean_chosen_forecast_brier,"always_pysteps_brier":macro.mean_always_pysteps_brier,"fixed_hybrid_brier":macro.mean_fixed_hybrid_brier,"stage8_final_scored":False,"promotable_model_trained":False}
    (ROOT/"stage9_decision.json").write_text(json.dumps(conclusion,indent=2))
    manifest=json.loads((ROOT/"stage9_diagnostic_manifest.json").read_text());manifest["outputs"]={p.name:sha256_file(p) for p in ROOT.glob("*.csv")};manifest["decision_sha256"]=sha256_file(ROOT/"stage9_decision.json");(ROOT/"stage9_diagnostic_manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
