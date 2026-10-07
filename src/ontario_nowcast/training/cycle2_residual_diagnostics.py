"""Post-selection DEV diagnostics for the frozen Cycle-2 Residual V1 checkpoint."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter
from .cycle2_residual import ROOT, ResidualCacheDataset, make_model, sha256_file


def _fss18(obs,pred):
    a=uniform_filter(obs.astype(np.float32),size=(1,19,19));b=uniform_filter(pred.astype(np.float32),size=(1,19,19))
    d=np.square(a).sum()+np.square(b).sum();return float(1-np.square(a-b).sum()/d) if d else 1.


def _first(x):
    p=x[:-1]&x[1:];found=p.any(0);return np.where(found,(p.argmax(0)+1)*6,np.nan)


def compact(truth,probability,valid,threshold):
    mask=valid.astype(bool);obs=(truth>.1)&mask;pred=(probability>=threshold)&mask
    h=(obs&pred).sum();m=(obs&~pred).sum();f=(~obs&pred&mask).sum();den=2*h+m+f
    ot=_first(obs);pt=_first(pred);paired=np.isfinite(ot)&np.isfinite(pt)
    return {"f1":float(2*h/den) if den else np.nan,"fss18":_fss18(obs,pred),
            "onset_mae":float(np.mean(np.abs(pt[paired]-ot[paired]))) if paired.any() else np.nan}


def run():
    selected=json.loads((ROOT/"selected_threshold.json").read_text())["selected_threshold"]
    manifest=pd.read_csv(ROOT/"cache_manifest.csv");dev=manifest[manifest.split.eq("new_dev")].reset_index(drop=True)
    predictions=pd.read_csv(ROOT/"dev_prediction_manifest.csv")[["row_id","prediction_path"]];dev=dev.merge(predictions,on="row_id")
    # Failure taxonomy uses only frozen PySTEPS/observations and never changes selection.
    labels=[];blind=[]
    for r in dev.itertuples(index=False):
        with np.load(r.cache_path) as z,np.load(r.prediction_path) as p:
            truth=z["target"].astype(np.float32);base=z["pysteps"].astype(np.float32);hist=z["history"].astype(np.float32);valid=z["target_validity"].astype(bool)
            obs=(truth>.1)&valid;bp=(base>.1)&valid;rp=(p["probability"].astype(np.float32)>=selected)&valid
            ow=obs.mean();bw=bp.mean();
            if r.row_type=="clean_initiation" and obs.any() and not bp.any(): cat="no initiation signal"
            elif ow>bw+.02: cat="growth underprediction"
            elif bw>ow+.02: cat="decay persistence"
            elif _fss18(obs,bp)>(2*(obs&bp).sum()/max(2*(obs&bp).sum()+(obs&~bp).sum()+(~obs&bp&valid).sum(),1))+.12: cat="displacement"
            else: cat="intensity error"
            for name,pred,rate in (("PySTEPS",bp.astype(float),base),("ResidualV1",p["probability"].astype(np.float32),p["corrected_rate"].astype(np.float32))):
                met=compact(truth,pred,valid,.5 if name=="PySTEPS" else selected)
                met.update({"row_id":r.row_id,"system_id":r.system_id,"category":cat,"model":name,
                            "rate_mae":float(np.mean(np.abs(rate[valid]-truth[valid])))});labels.append(met)
            # Pixel-level radar-blind definition: dry throughout history, PySTEPS dry, future persistent wet.
            eligible=(hist<=.1).all(0)&~(base>.1).any(0)&np.isfinite(_first(obs)) if r.row_type=="clean_initiation" else np.zeros(obs.shape[1:],bool)
            predicted=np.isfinite(_first(rp));dry=~np.isfinite(_first(obs))
            blind.append({"row_id":r.row_id,"system_id":r.system_id,"eligible_locations":int(eligible.sum()),
                          "detected_locations":int((eligible&predicted).sum()),"false_area_pixels":int((dry&predicted).sum()),
                          "valid_dry_pixels":int(dry.sum())})
    lab=pd.DataFrame(labels);lab.to_csv(ROOT/"failure_category_by_row.csv",index=False)
    lab.groupby(["category","model"],as_index=False).agg(cases=("row_id","nunique"),f1=("f1","mean"),fss18=("fss18","mean"),rate_mae=("rate_mae","mean")).to_csv(ROOT/"failure_category_summary.csv",index=False)
    blind=pd.DataFrame(blind);blind.to_csv(ROOT/"radar_blind_initiation_by_row.csv",index=False)
    be=blind.groupby("system_id",as_index=False).sum(numeric_only=True);eligible=blind.eligible_locations.sum()
    blind_summary={"eligible_locations":int(eligible),"eligible_rows":int((blind.eligible_locations>0).sum()),
                   "independent_systems":int((be.eligible_locations>0).sum()),"detection_fraction":float(blind.detected_locations.sum()/eligible) if eligible else np.nan,
                   "false_area_fraction":float(blind.false_area_pixels.sum()/blind.valid_dry_pixels.sum()),
                   "systems_with_any_detection":int(((be.eligible_locations>0)&(be.detected_locations>0)).sum())}
    (ROOT/"radar_blind_initiation_summary.json").write_text(json.dumps(blind_summary,indent=2)+"\n")
    # Destruction tests: same checkpoint and selected threshold, no retraining.
    norm=json.loads((ROOT/"normalization.json").read_text());model=make_model(norm)
    checkpoint=torch.load(ROOT/"checkpoint_best.pt",map_location="cpu",weights_only=False);model.load_state_dict(checkpoint["model"]);model.eval()
    dataset=ResidualCacheDataset(dev);records=[]
    with torch.no_grad():
        for i in range(len(dataset)):
            x=dataset[i];h=x["history"][None];ps=x["pysteps"][None];motion=x["motion"][None];v=x["validity"][None]
            variants={"intact":(h,ps,motion,v),"zero_pysteps":(h,torch.zeros_like(ps),motion,v),
                      "latest_history_only":(torch.cat((torch.zeros_like(h[:,:-1]),h[:,-1:] ),1),ps,motion,torch.cat((torch.zeros_like(v[:,:-1]),v[:,-1:]),1)),
                      "zero_motion":(h,ps,torch.zeros_like(motion),v)}
            intact_probability=None
            for name,args in variants.items():
                out=model(*args);prob=torch.sigmoid(out["occurrence_logits"])[0].numpy();met=compact(x["target"].numpy(),prob,x["target_validity"].numpy(),selected)
                if name=="intact": intact_probability=prob
                met.update({"variant":name,"row_id":x["row_id"],"system_id":x["system_id"]});records.append(met)
            # Zeroing the intensity residual leaves the separately learned occurrence head unchanged by definition.
            met=compact(x["target"].numpy(),intact_probability,x["target_validity"].numpy(),selected)
            met.update({"variant":"zero_intensity_residual","row_id":x["row_id"],"system_id":x["system_id"]});records.append(met)
            if (i+1)%50==0: print(f"DESTRUCTION {i+1}/{len(dataset)}",flush=True)
    dr=pd.DataFrame(records);dr.to_csv(ROOT/"destruction_metrics_by_row.csv",index=False)
    de=dr.groupby(["variant","system_id"],as_index=False).mean(numeric_only=True);dm=de.groupby("variant",as_index=False).mean(numeric_only=True)
    intact=dm[dm.variant.eq("intact")].iloc[0]
    for metric in ("f1","fss18","onset_mae"):dm[f"delta_{metric}_vs_intact"]=dm[metric]-intact[metric]
    dm.to_csv(ROOT/"destruction_summary.csv",index=False)
    return {"failure_categories":sorted(lab.category.unique()),"radar_blind":blind_summary,"destruction":dm.to_dict("records")}
