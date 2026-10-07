"""Frozen DEV-only evaluation for Cycle-2 PySTEPS Residual V1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter

from .cycle2_residual import ROOT, ResidualCacheDataset, make_model, sha256_file

LEADS = {30: 4, 60: 9, 90: 14, 120: 19}
THRESHOLDS = np.round(np.arange(.10, .901, .05), 2)


def _ratio(a, b): return float(a / b) if b else float("nan")


def _persistent_time(binary):
    x = binary[:-1] & binary[1:]
    any_x = x.any(0)
    return np.where(any_x, (x.argmax(0) + 1) * 6, np.nan)


def _cessation_time(binary):
    dry = ~binary
    x = dry[:-1] & dry[1:]
    any_x = x.any(0)
    return np.where(any_x, (x.argmax(0) + 1) * 6, np.nan)


def _auc(y, p):
    y = y.astype(bool); n1 = y.sum(); n0 = len(y)-n1
    if not n1 or not n0: return np.nan
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n1*(n1+1)/2)/(n1*n0))


def row_metrics(truth, probability, rate, valid, threshold, *, row_id, system_id, row_type, model):
    mask=valid.astype(bool); obs=(truth>.1)&mask; pred=(probability>=threshold)&mask
    h=int((obs&pred).sum()); miss=int((obs&~pred).sum()); fa=int((~obs&pred&mask).sum())
    out={"row_id":row_id,"system_id":system_id,"row_type":row_type,"model":model,
         "hits":h,"misses":miss,"false_alarms":fa,"csi":_ratio(h,h+miss+fa),
         "pod":_ratio(h,h+miss),"far":_ratio(fa,h+fa),"f1":_ratio(2*h,2*h+fa+miss),
         "brier":float(np.mean(np.square(probability[mask]-obs[mask]))),
         "auc":_auc(obs[mask].ravel(),probability[mask].ravel()),
         "wet_area":float(pred[mask].mean()),"mean_probability":float(probability[mask].mean()),
         "max_probability":float(probability[mask].max()),
         "rate_mae":float(np.mean(np.abs(rate[mask]-truth[mask]))),
         "log_rate_mae":float(np.mean(np.abs(np.log1p(rate[mask])-np.log1p(truth[mask]))))}
    heavy=(truth>5)&mask;out["heavy_rain_conditional_error"]=float(np.mean(np.abs(rate[heavy]-truth[heavy]))) if heavy.any() else np.nan
    for km,kernel in ((6,7),(18,19),(36,37)):
        of=uniform_filter(obs.astype(np.float32),size=(1,kernel,kernel));pf=uniform_filter(pred.astype(np.float32),size=(1,kernel,kernel))
        den=np.square(of).sum()+np.square(pf).sum();out[f"fss{km}"]=float(1-np.square(of-pf).sum()/den) if den else 1.
    ot=_persistent_time(obs);pt=_persistent_time(pred);paired=np.isfinite(ot)&np.isfinite(pt)
    out["onset_mae"]=float(np.mean(np.abs(pt[paired]-ot[paired]))) if paired.any() else np.nan
    out["onset_bias"]=float(np.median(pt[paired]-ot[paired])) if paired.any() else np.nan
    oc=_cessation_time(obs);pc=_cessation_time(pred);cp=np.isfinite(oc)&np.isfinite(pc)
    out["cessation_mae"]=float(np.mean(np.abs(pc[cp]-oc[cp]))) if cp.any() else np.nan
    out["false_initiation"]=float(np.isfinite(pt).any()) if row_type=="hard_negative" else np.nan
    out["radar_blind_pixels"]=int((((truth>.1).any(0))&~((rate>.1).any(0))).sum()) if model=="PySTEPS" else np.nan
    for minute,index in LEADS.items():
        leadmask=mask[index];yo=obs[index]&leadmask;yp=pred[index]&leadmask
        hh=int((yo&yp).sum());mm=int((yo&~yp).sum());ff=int((~yo&yp&leadmask).sum())
        out[f"f1_{minute}"]=_ratio(2*hh,2*hh+ff+mm);out[f"csi_{minute}"]=_ratio(hh,hh+ff+mm)
        out[f"pod_{minute}"]=_ratio(hh,hh+mm);out[f"far_{minute}"]=_ratio(ff,hh+ff)
        of=uniform_filter(yo.astype(np.float32),size=19);pf=uniform_filter(yp.astype(np.float32),size=19)
        den=np.square(of).sum()+np.square(pf).sum();out[f"fss18_{minute}"]=float(1-np.square(of-pf).sum()/den) if den else 1.
        out[f"rate_mae_{minute}"]=float(np.mean(np.abs(rate[index][leadmask]-truth[index][leadmask])))
        out[f"brier_{minute}"]=float(np.mean(np.square(probability[index][leadmask]-yo[leadmask])))
    return out


def _load_best():
    norm=json.loads((ROOT/"normalization.json").read_text());model=make_model(norm)
    checkpoint=torch.load(ROOT/"checkpoint_best.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["model"]);model.eval();return model,checkpoint


def generate_predictions():
    manifest=pd.read_csv(ROOT/"cache_manifest.csv")
    if (manifest.split=="new_final").any(): raise PermissionError("FINAL cache entry found")
    dev=manifest[manifest.split.eq("new_dev")].reset_index(drop=True);dataset=ResidualCacheDataset(dev)
    model,checkpoint=_load_best();outdir=ROOT/"dev_predictions";outdir.mkdir(parents=True,exist_ok=True)
    existing=ROOT/"dev_prediction_manifest.csv"
    if existing.exists():
        prior=pd.read_csv(existing).rename(columns={"output_sha256":"prediction_sha256"})
        joined=dev.merge(prior,on="row_id",how="inner")
        valid=(len(joined)==len(dev) and all(Path(r.prediction_path).exists() and
               sha256_file(Path(r.prediction_path))==r.prediction_sha256 for r in joined.itertuples(index=False)))
        if valid:
            print(f"REUSED {len(joined)}/381 HASH-VERIFIED DEV PREDICTIONS",flush=True)
            return joined,checkpoint
    records=[]
    with torch.no_grad():
        for i in range(len(dataset)):
            item=dataset[i]
            output=model(item["history"][None],item["pysteps"][None],item["motion"][None],item["validity"][None])
            path=outdir/f"{item['row_id']}.npz"
            np.savez_compressed(path,probability=torch.sigmoid(output["occurrence_logits"])[0].numpy().astype(np.float16),
                corrected_rate=output["corrected_rate"][0].numpy().astype(np.float16))
            records.append({"row_id":item["row_id"],"prediction_path":str(path),"output_sha256":sha256_file(path)})
            if (i+1)%50==0: print(f"PREDICTIONS {i+1}/{len(dataset)}",flush=True)
    frame=pd.DataFrame(records);frame.to_csv(ROOT/"dev_prediction_manifest.csv",index=False)
    return dev.merge(frame,on="row_id"),checkpoint


def _candidate_metrics(dev, threshold):
    rows=[]
    for r in dev.itertuples(index=False):
        with np.load(r.cache_path) as z, np.load(r.prediction_path) as p:
            rows.append(row_metrics(z["target"].astype(np.float32),p["probability"].astype(np.float32),p["corrected_rate"].astype(np.float32),z["target_validity"],threshold,row_id=r.row_id,system_id=r.system_id,row_type=r.row_type,model="ResidualV1"))
    x=pd.DataFrame(rows);positive=x[x.row_type.ne("hard_negative")];events=positive.groupby("system_id").mean(numeric_only=True)
    neg=x[x.row_type.eq("hard_negative")]
    # Comparator dry-area increase is measured on the exact same negative pixels.
    baseline_wet=[]
    for r in dev[dev.row_type.eq("hard_negative")].itertuples(index=False):
        with np.load(r.cache_path) as z: baseline_wet.append(float((z["pysteps"]>.1).mean()))
    return {"threshold":threshold,"csi":events.csi.mean(),"pod":events.pod.mean(),
            "far":events.far.mean(),"f1":events.f1.mean(),"fss6":events.fss6.mean(),
            "fss18":events.fss18.mean(),"fss36":events.fss36.mean(),
            "onset_mae":events.onset_mae.mean(),"onset_bias":events.onset_bias.mean(),
            "hard_negative_wet_area":neg.wet_area.mean(),"pysteps_hard_negative_wet_area":np.mean(baseline_wet),
            "hard_negative_wet_area_increase":neg.wet_area.mean()-np.mean(baseline_wet),
            "hard_negative_false_initiation":neg.false_initiation.mean()}


def select_threshold(dev):
    candidates=pd.DataFrame([_candidate_metrics(dev,float(t)) for t in THRESHOLDS])
    candidates["constraints_pass"]=(candidates.hard_negative_wet_area_increase<=.01)&(candidates.hard_negative_false_initiation<=.40)
    eligible=candidates[candidates.constraints_pass]
    if eligible.empty: selected=None
    else:selected=float(eligible.sort_values(["f1","fss18","far","onset_mae","threshold"],ascending=[False,False,True,True,False]).iloc[0].threshold)
    candidates["selected"] = candidates.threshold.eq(selected) if selected is not None else False
    candidates.to_csv(ROOT/"dev_threshold_sweep.csv",index=False)
    (ROOT/"selected_threshold.json").write_text(json.dumps({"selected_threshold":selected,"feasible_candidates":int(len(eligible)),"rule":"maximize event-macro F1 subject to frozen hard-negative constraints; tie FSS18, FAR, onset MAE, higher threshold"},indent=2)+"\n")
    return selected,candidates


def evaluate():
    dev,checkpoint=generate_predictions();threshold,candidates=select_threshold(dev)
    if threshold is None:
        summary={"status":"NO_FEASIBLE_OPERATING_THRESHOLD","best_epoch":checkpoint["epoch"],"sealed_final_accessed":False}
        (ROOT/"evaluation_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary
    rows=[]
    for r in dev.itertuples(index=False):
        with np.load(r.cache_path) as z,np.load(r.prediction_path) as p:
            truth=z["target"].astype(np.float32);valid=z["target_validity"]
            rows.append(row_metrics(truth,p["probability"].astype(np.float32),p["corrected_rate"].astype(np.float32),valid,threshold,row_id=r.row_id,system_id=r.system_id,row_type=r.row_type,model="ResidualV1"))
            base=z["pysteps"].astype(np.float32);rows.append(row_metrics(truth,(base>.1).astype(np.float32),base,valid,.5,row_id=r.row_id,system_id=r.system_id,row_type=r.row_type,model="PySTEPS"))
    metrics=pd.DataFrame(rows);metrics.to_csv(ROOT/"dev_metrics_by_row.csv",index=False)
    event=metrics.groupby(["model","system_id"],as_index=False).mean(numeric_only=True);event.to_csv(ROOT/"dev_metrics_by_system.csv",index=False)
    macro=event.groupby("model",as_index=False).mean(numeric_only=True);macro.to_csv(ROOT/"dev_event_macro_metrics.csv",index=False)
    metrics.groupby(["model","row_type"],as_index=False).mean(numeric_only=True).to_csv(ROOT/"dev_metrics_by_row_type.csv",index=False)
    leadcols=["model"]+[f"{metric}_{m}" for m in LEADS for metric in ("f1","csi","pod","far","fss18","rate_mae","brier")];macro[leadcols].to_csv(ROOT/"dev_lead_metrics.csv",index=False)
    a=event[event.model.eq("ResidualV1")].set_index("system_id");b=event[event.model.eq("PySTEPS")].set_index("system_id")
    delta=a.f1-b.f1;am=macro.set_index("model").loc["ResidualV1"];bm=macro.set_index("model").loc["PySTEPS"]
    chosen=candidates[candidates.threshold.eq(threshold)].iloc[0]
    gates={"F1_gain_at_least_0.01":bool(am.f1>=bm.f1+.01),"FSS18_not_decreased":bool(am.fss18>=bm.fss18),
           "onset_MAE_within_3min":bool(am.onset_mae<=bm.onset_mae+3),"systems_noninferior_at_least_7":bool((delta>=-.005).sum()>=7),
           "systems_strictly_better_at_least_5":bool((delta>0).sum()>=5),"hard_negative_wet_area_increase_le_0.01":bool(chosen.hard_negative_wet_area_increase<=.01),
           "hard_negative_false_initiation_le_0.40":bool(chosen.hard_negative_false_initiation<=.40),"Brier_degradation_le_0.005":bool(am.brier<=bm.brier+.005)}
    summary={"status":"DEV_EVALUATION_COMPLETE","best_epoch":int(checkpoint["epoch"]),"checkpoint_sha256":sha256_file(ROOT/"checkpoint_best.pt"),
             "selected_threshold":threshold,"macro":macro.set_index("model").to_dict("index"),"promotion_gates":gates,
             "promoted":bool(all(gates.values())),"systems_strictly_better":int((delta>0).sum()),"systems_noninferior":int((delta>=-.005).sum()),"sealed_final_accessed":False}
    (ROOT/"evaluation_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=True)+"\n")
    return summary
