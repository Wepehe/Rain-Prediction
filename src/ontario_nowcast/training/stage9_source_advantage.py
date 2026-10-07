"""Stage 9 diagnostic-only PySTEPS/HRRR source-advantage analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

from ..data.manifest import sha256_file
from ..models.baselines import pysteps_deterministic_extrapolation


OUT = Path("artifacts/stage_9/source_advantage")
FEATURES = [
    "lead_fraction", "current_radar_wet_fraction", "recent_radar_growth",
    "pysteps_wet_fraction", "a_plus_mean_entropy", "hrrr_wet_fraction",
    "hrrr_pysteps_disagreement_fraction", "hrrr_radar_support_fraction",
]


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    n1, n0 = int(y.sum()), int((1-y).sum())
    if not n1 or not n0:
        return float("nan")
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_logistic(x: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float = 1.0):
    mean, std = x.mean(0), x.std(0); std[std == 0] = 1
    z = (x - mean) / std; design = np.c_[np.ones(len(z)), z]
    counts = pd.Series(groups).value_counts(); w = np.asarray([1 / counts[g] for g in groups]); w /= w.sum()
    def fg(beta):
        eta = np.clip(design @ beta, -30, 30); p = 1 / (1 + np.exp(-eta))
        loss = -np.sum(w * (y*np.log(np.clip(p,1e-8,1)) + (1-y)*np.log(np.clip(1-p,1e-8,1)))) + lam*np.sum(beta[1:]**2)
        grad = design.T @ (w*(p-y)); grad[1:] += 2*lam*beta[1:]
        return loss, grad
    res = minimize(lambda b: fg(b), np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    return mean, std, res


def _predict(x, mean, std, beta):
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(x)), (x-mean)/std] @ beta, -30, 30)))


def _record(collection, row_id, event, object_id, issue, role, cohort, j, feat, a, py, h, y, valid, py_rate=None, target_rate=None):
    q = valid.astype(bool)
    if not np.any(q): return None
    aa, pp, hh, yy = a[q], py[q], h[q], y[q]
    mix = .5*pp + .5*hh
    b_py = np.mean((pp-yy)**2); b_h = np.mean((hh-yy)**2); b_mix = np.mean((mix-yy)**2)
    e_py = np.mean(np.abs(pp-yy)); e_h = np.mean(np.abs(hh-yy)); e_mix = np.mean(np.abs(mix-yy))
    d_h, d_mix = b_py-b_h, b_py-b_mix
    eps=.005
    if d_h > eps and d_mix > eps: label="HRRR_and_combination_better"
    elif d_h > eps: label="HRRR_clearly_better"
    elif d_mix > eps: label="combination_better"
    elif d_h < -eps and d_mix < -eps: label="PySTEPS_clearly_better"
    else: label="approximately_tied"
    truth_wet=yy>=.5; py_wet=pp>=.5; h_wet=hh>=.5
    inter=np.logical_and(py_wet,truth_wet).sum(); union=np.logical_or(py_wet,truth_wet).sum(); iou=inter/union if union else 1.
    if truth_wet.any() and not py_wet.any(): failure="no_initiation_signal"
    elif py_wet.mean() > truth_wet.mean()*1.5 and feat[1] > 0: failure="decay_persistence"
    elif truth_wet.mean() > py_wet.mean()*1.5: failure="growth_underprediction"
    elif truth_wet.any() and py_wet.any() and iou < .2: failure="displacement"
    elif py_rate is not None and target_rate is not None and np.nanmean(np.abs(py_rate[q]-target_rate[q])) > 1: failure="intensity_error"
    else: failure="other_or_adequate"
    h_inter=np.logical_and(h_wet,truth_wet).sum(); h_union=np.logical_or(h_wet,truth_wet).sum(); h_iou=h_inter/h_union if h_union else 1.
    if h_wet.any() and not truth_wet.any(): hfail="false_precipitation"
    elif truth_wet.any() and not h_wet.any(): hfail="missed_precipitation"
    elif truth_wet.any() and h_wet.any() and h_iou < .2: hfail="spatial_displacement"
    elif h_wet.mean() > truth_wet.mean()*1.5: hfail="broad_area_poor_localization"
    elif j < 10 and h_wet.any() and not truth_wet.any(): hfail="hourly_signal_too_early"
    else: hfail="other_or_adequate"
    return {"collection":collection,"row_id":row_id,"event_id":event,"object_id":object_id,"issue_time_utc":issue,"role":role,"cohort":cohort,"lead_minutes":6*(j+1),
            **dict(zip(FEATURES,feat)),"brier_pysteps":b_py,"brier_hrrr":b_h,"brier_pysteps_hrrr_hybrid":b_mix,
            "delta_brier_hrrr_vs_pysteps":d_h,"delta_brier_hybrid_vs_pysteps":d_mix,"delta_occurrence_error_hrrr_vs_pysteps":e_py-e_h,
            "delta_occurrence_error_hybrid_vs_pysteps":e_py-e_mix,"oracle_label":label,"hrrr_beats_pysteps":int(d_h>0),"hybrid_beats_pysteps":int(d_mix>0),
            "pysteps_failure_category":failure,"hrrr_failure_category":hfail}


def _stage8_records():
    m=pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv")
    cohorts=pd.read_csv("artifacts/stage_8/development_features/stage8_development_radar_cohorts.csv").set_index("row_id")
    rows=[]
    for r in m.itertuples(index=False):
        z=np.load(r.cache_path); f=z["features"].astype(float); a=z["a_plus_probability"].astype(float); py_rate=z["pysteps_rate"].astype(float); py=(py_rate>=.1).astype(float); h=z["hrrr_occurrence"].astype(float); y=z["target_occurrence"].astype(float); v=z["target_valid"].astype(bool); tr=z["target_rate"].astype(float)
        growth=0.; cohort="hard_negative" if r.role=="hard_negative" else ("RADAR_LIMITED_INITIATION" if r.row_id in cohorts.index and bool(cohorts.loc[r.row_id,"radar_limited_initiation"]) else "initiation_with_radar_precursor")
        for j in range(20):
            feat=np.array([f[j,0],f[j,1],growth,f[j,4],f[j,3],f[j,5],np.mean((py[j]>=.5)!=(h[j]>=.5)),np.mean((py[j]>=.5)[h[j]>=.5]) if h[j].any() else 0.])
            rec=_record("stage8_nested_development",r.row_id,r.independent_system_group,r.row_id.split("__")[-1],r.issue_time_utc,r.role,cohort,j,feat,a[j],py[j],h[j],y[j],v[j],py_rate[j],tr[j])
            if rec: rows.append(rec)
    return rows


def _stage6_records():
    cache=pd.read_csv("artifacts/stage_6/stage6_cache_manifest.csv")
    hold=pd.read_csv("artifacts/stage_4c/frozen_holdout/stage4c_final_holdout_manifest.csv")
    table=cache.merge(hold,left_on=["event_id","object_id"],right_on=["source_event_id","object_id"],suffixes=("","_frozen"),validate="one_to_one")
    rows=[]
    for n,r in enumerate(table.itertuples(index=False),1):
        s=np.load(r.tensor_path); c=np.load(r.stage6_cache_path); radar=s["radar_rate_mm_hr"].astype(float); target=s["target_rate_mm_hr"].astype(float); valid=s["target_valid_mask"].astype(bool)
        a=1/(1+np.exp(-c["aplus_logits"][:,0].astype(float))); py_rate=pysteps_deterministic_extrapolation(radar[-3:],20).astype(float); py=(py_rate>=.1).astype(float); hourly=c["hrrr_apcp_hourly"].astype(float); h=np.stack([hourly[0]>=.1]*10+[hourly[1]>=.1]*10).astype(float)
        cur=float(np.nanmean(radar[-1]>=.1)); growth=float(np.nanmean(radar[-1]>=.1)-np.nanmean(radar[-3]>=.1)); cohort=("hard_negative" if r.row_role=="hard_negative" else "heavy_rain" if r.row_role=="heavy_rain_generalization" else "RADAR_POOR_INITIATION_V2" if bool(r.radar_poor_initiation_v2) else "RADAR_LIMITED_INITIATION" if bool(r.radar_limited_initiation) else "initiation_with_radar_precursor")
        issue=r.representative_issue_time_utc
        for j in range(20):
            pc=np.clip(a[j],1e-6,1-1e-6); feat=np.array([(j+1)/20,cur,growth,(py[j]>=.5).mean(),np.mean(-pc*np.log(pc)-(1-pc)*np.log(1-pc)),h[j].mean(),np.mean((py[j]>=.5)!=(h[j]>=.5)),np.mean((py[j]>=.5)[h[j]>=.5]) if h[j].any() else 0.])
            rec=_record("stage6_2023_consumed",r.identity,r.event_id,r.object_id,issue,r.row_role,cohort,j,feat,a[j],py[j],h[j],target[j]>=.1,valid[j],py_rate[j],target[j])
            if rec: rows.append(rec)
        if n%10==0: print(f"[stage9] Stage 6 rows {n}/{len(table)}",flush=True)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    oracle=pd.DataFrame(_stage8_records()+_stage6_records()); oracle.to_csv(OUT/"oracle_source_advantage_by_row_lead.csv",index=False)
    positive=oracle[~oracle.role.isin(["hard_negative"])].copy()
    positive.groupby(["collection","event_id"],as_index=False).agg(rows=("row_id","nunique"),mean_delta_brier_hrrr=("delta_brier_hrrr_vs_pysteps","mean"),mean_delta_brier_hybrid=("delta_brier_hybrid_vs_pysteps","mean"),hrrr_win_fraction=("hrrr_beats_pysteps","mean"),hybrid_win_fraction=("hybrid_beats_pysteps","mean")).to_csv(OUT/"oracle_by_event.csv",index=False)
    positive.groupby(["collection","cohort"],as_index=False).agg(row_leads=("row_id","size"),events=("event_id","nunique"),mean_delta_brier_hrrr=("delta_brier_hrrr_vs_pysteps","mean"),mean_delta_brier_hybrid=("delta_brier_hybrid_vs_pysteps","mean"),hrrr_win_fraction=("hrrr_beats_pysteps","mean")).to_csv(OUT/"oracle_by_cohort.csv",index=False)
    positive.groupby(["collection","oracle_label"],as_index=False).size().to_csv(OUT/"oracle_label_distribution.csv",index=False)
    positive.groupby(["collection","pysteps_failure_category"],as_index=False).agg(count=("row_id","size"),hrrr_corrects=("hrrr_beats_pysteps","mean"),hybrid_corrects=("hybrid_beats_pysteps","mean"),mean_delta_brier=("delta_brier_hrrr_vs_pysteps","mean")).to_csv(OUT/"pysteps_failure_analysis.csv",index=False)
    positive.groupby(["collection","hrrr_failure_category"],as_index=False).agg(count=("row_id","size"),hrrr_win_fraction=("hrrr_beats_pysteps","mean"),mean_delta_brier=("delta_brier_hrrr_vs_pysteps","mean")).to_csv(OUT/"hrrr_failure_analysis.csv",index=False)

    # Exploratory grouped LOEO diagnostic. Stage-8 development is used because its seven systems permit event-level replication.
    d=positive[positive.collection.eq("stage8_nested_development")].copy(); x=d[FEATURES].to_numpy(float); y=d.hrrr_beats_pysteps.to_numpy(float); groups=d.event_id.to_numpy(); pred=np.zeros(len(d)); coefs=[]
    for hold in sorted(np.unique(groups)):
        tr=groups!=hold; mean,std,res=_fit_logistic(x[tr],y[tr],groups[tr]); pred[~tr]=_predict(x[~tr],mean,std,res.x)
        coefs.append({"held_out_event":hold,"intercept":res.x[0],"success":res.success,**dict(zip(FEATURES,res.x[1:]))})
    d["advantage_probability_oof"]=pred; d.to_csv(OUT/"advantage_predictor_oof.csv",index=False); pd.DataFrame(coefs).to_csv(OUT/"advantage_predictor_coefficients.csv",index=False)
    ev=[]
    global_rate=float(y.mean())
    for g,t in d.groupby("event_id"):
        yy=t.hrrr_beats_pysteps.to_numpy(); pp=t.advantage_probability_oof.to_numpy(); choose=pp>=.5
        chosen=np.where(choose,t.brier_hrrr,t.brier_pysteps)
        ev.append({"event_id":g,"samples":len(t),"positive_rate":yy.mean(),"auc":_auc(yy,pp),"accuracy":np.mean((pp>=.5)==yy),"brier":np.mean((pp-yy)**2),"chosen_forecast_brier":chosen.mean(),"always_pysteps_brier":t.brier_pysteps.mean(),"always_hrrr_brier":t.brier_hrrr.mean(),"fixed_hybrid_brier":t.brier_pysteps_hrrr_hybrid.mean(),"event_base_rate_brier":np.mean((global_rate-yy)**2)})
    ev=pd.DataFrame(ev);ev.to_csv(OUT/"advantage_predictor_by_event.csv",index=False)
    pd.DataFrame([{"events":len(ev),**{f"mean_{c}":ev[c].mean() for c in ["auc","accuracy","brier","chosen_forecast_brier","always_pysteps_brier","always_hrrr_brier","fixed_hybrid_brier","event_base_rate_brier"]}}]).to_csv(OUT/"advantage_predictor_event_macro.csv",index=False)

    # Stage-8 post-mortem: association, sign stability and variance decomposition.
    assoc=[]; variance=[]
    for f in FEATURES:
        rho,pv=spearmanr(d[f],d.delta_brier_hrrr_vs_pysteps,nan_policy="omit"); signs=[]
        for _,t in d.groupby("event_id"):
            rr=spearmanr(t[f],t.delta_brier_hrrr_vs_pysteps,nan_policy="omit").statistic; signs.append(rr)
        event_means=d.groupby("event_id")[f].mean(); between=float(event_means.var()); within=float(d.groupby("event_id")[f].var().mean())
        assoc.append({"feature":f,"spearman_all":rho,"p_value_descriptive":pv,"positive_event_associations":sum(np.isfinite(signs)&(np.asarray(signs)>0)),"negative_event_associations":sum(np.isfinite(signs)&(np.asarray(signs)<0))})
        variance.append({"feature":f,"between_event_variance":between,"within_event_variance":within,"between_fraction":between/(between+within) if between+within else np.nan})
    pd.DataFrame(assoc).to_csv(OUT/"feature_oracle_associations.csv",index=False);pd.DataFrame(variance).to_csv(OUT/"feature_variance_decomposition.csv",index=False)
    alignment=d.groupby("event_id").agg(delta_brier=("delta_brier_hybrid_vs_pysteps","mean"),hybrid_win_fraction=("hybrid_beats_pysteps","mean")).reset_index(); alignment.to_csv(OUT/"brier_initiation_alignment_by_event.csv",index=False)

    # Cross-collection hierarchy from the immutable evaluation tables.
    s6=pd.read_csv("artifacts/stage_6/evaluation/all_initiation_by_event.csv"); s6["collection"]="Stage 6 consumed 2023"
    s8=pd.read_csv("artifacts/stage_8/nested_development/comparator_by_event_metrics.csv").rename(columns={"independent_system_group":"event_id"});s8["collection"]="Stage 8 nested development"
    s6["model"]=s6.model.replace({"hybrid":"Stage6_hybrid"}); hierarchy=pd.concat([s6,s8],ignore_index=True,sort=False);hierarchy.to_csv(OUT/"baseline_hierarchy_by_event.csv",index=False)
    stability=hierarchy.groupby(["collection","model"],as_index=False).agg(events=("event_id","nunique"),f1_mean=("f1","mean"),f1_std=("f1","std"),f1_min=("f1","min"),f1_max=("f1","max"),precision_mean=("precision","mean"),recall_mean=("recall","mean"),false_initiation_mean=("false_initiation_fraction","mean"),onset_mae_mean=("onset_mae_minutes","mean"),brier_mean=("brier_score","mean"));stability["f1_range"]=stability.f1_max-stability.f1_min;stability.to_csv(OUT/"baseline_cross_event_stability.csv",index=False)
    manifest={"status":"DIAGNOSTIC COMPLETE","promotable_model_trained":False,"sealed_stage8_final_scored":False,"rows":int(oracle.row_id.nunique()),"row_leads":len(oracle),"events":int(oracle.event_id.nunique()),"outputs":{p.name:sha256_file(p) for p in OUT.glob("*.csv")}}
    (OUT/"stage9_diagnostic_manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
