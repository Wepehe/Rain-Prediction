"""One-time frozen Cycle-2 FINAL evaluation.

Generation and scoring are deliberately separate.  ``score_final`` refuses to
run until all 353 row artifacts and their hashes have been sealed.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from ..models.pysteps_residual_unet import PySTEPSResidualUNetV1, count_parameters
from ..sample_evaluation import _downsample_max
from .cycle2_residual import deterministic_pysteps_with_motion, sha256_file
from .cycle2_residual_eval import row_metrics

ROOT=Path("artifacts/cycle2/residual_v1")
OUT=ROOT/"final_evaluation"
SEALED=Path("artifacts/cycle2/data/final_manifest.json")
SPLIT=Path("artifacts/cycle2/data/frozen_system_split.csv")
EXPECTED={"final_manifest":"d71dd81da4663c98c84858a6890ee824922033b68140fb38cedaea7928c9b0c3",
          "checkpoint":"b536ca16daaaf8904b9c326ed00c01169105d764c2072249092a9259ba9a22c5",
          "normalization":"23dc3b31edad1247fc77abd4ec7d44688cdd79e21b3d0f832260130df36c192c",
          "configuration":"13456f0e6a940f3a8a41f68e7c9e7f7efdf4eaf5e24273bcc96f5b967250495b",
          "model_source":"2a84095e36009174ca662566b2158e8e9fb38e3e17fba280ceb8017e2f739ea2"}
PATHS={"final_manifest":SEALED,"checkpoint":ROOT/"checkpoint_best.pt","normalization":ROOT/"normalization.json",
       "configuration":Path("configs/cycle2_residual_v1.yaml"),"model_source":Path("src/ontario_nowcast/models/pysteps_residual_unet.py")}


def _json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=True)+"\n",encoding="utf-8");tmp.replace(path)


def _hash_arrays(*arrays):
    h=hashlib.sha256()
    for x in arrays:
        a=np.ascontiguousarray(x);h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def frozen_gate():
    checks={k:{"actual":sha256_file(p),"expected":EXPECTED[k]} for k,p in PATHS.items()}
    for value in checks.values():value["pass"]=value["actual"]==value["expected"]
    proc=json.loads((ROOT/"final_procedure_manifest.json").read_text())
    checks["threshold"]={"actual":proc["operating_threshold"],"expected":.35,"pass":proc["operating_threshold"]==.35}
    checks["parameters"]={"actual":count_parameters(PySTEPSResidualUNetV1()),"expected":3060440,"pass":count_parameters(PySTEPSResidualUNetV1())==3060440}
    if not all(x["pass"] for x in checks.values()):raise RuntimeError(f"frozen gate failed: {checks}")
    return checks


def final_rows():
    sealed=json.loads(SEALED.read_text());rows=pd.DataFrame(sealed["rows"]);systems=pd.DataFrame(sealed["systems"])
    paths=pd.read_csv(SPLIT)[["system_id","source_path","source_sha256","split"]]
    rows=rows.drop(columns=["source_sha256","split"],errors="ignore").merge(paths,on="system_id",validate="many_to_one")
    if len(systems)!=12 or len(rows)!=353 or rows.system_id.nunique()!=12:raise RuntimeError("FINAL count mismatch")
    counts=rows.row_type.value_counts().to_dict()
    if counts!={"active_precip":288,"clean_initiation":56,"hard_negative":9}:raise RuntimeError(f"row counts: {counts}")
    if set(rows.split)!={"new_final"}:raise RuntimeError("non-FINAL row in sealed manifest")
    nonfinal=pd.read_csv(Path("artifacts/cycle2/data/train_dev_rows.csv"))
    overlap=set(rows.system_id)&set(nonfinal.system_id)
    if overlap:raise RuntimeError(f"FINAL leakage into TRAIN/DEV: {overlap}")
    return rows,systems


def generate_final():
    checks=frozen_gate();rows,systems=final_rows()
    if (OUT/"GENERATION_COMPLETE.json").exists():raise RuntimeError("FINAL predictions already generated; one-time run cannot be repeated")
    OUT.mkdir(parents=True,exist_ok=True);tensor_dir=OUT/"row_tensors";tensor_dir.mkdir(exist_ok=True)
    normalization=json.loads((ROOT/"normalization.json").read_text())
    model=PySTEPSResidualUNetV1();model.set_normalization(rate_mean=normalization["rate_log1p_mean"],rate_std=normalization["rate_log1p_std"],motion_mean=normalization["motion_mean"],motion_std=normalization["motion_std"])
    ck=torch.load(ROOT/"checkpoint_best.pt",map_location="cpu",weights_only=False);model.load_state_dict(ck["model"]);model.eval()
    records=[];integrity=[]
    with torch.no_grad():
      for system,group in rows.groupby("system_id",sort=True):
        source=Path(group.source_path.iloc[0]);actual=sha256_file(source);expected=group.source_sha256.iloc[0]
        if actual!=expected:raise RuntimeError(f"source hash mismatch: {system}")
        with np.load(source) as z:
            times=pd.to_datetime(z["times"],utc=True);rates=_downsample_max(z["rate_mm_hr"],2)
        system_records=[]
        for r in group.itertuples(index=False):
            anchor=int(times.get_indexer([pd.Timestamp(r.issue_time_utc)])[0])
            if anchor<9 or anchor+20>=len(times):raise RuntimeError(f"window bounds: {r.row_id}")
            window=times[anchor-9:anchor+21]
            if len(window)!=30 or not np.all((window[1:]-window[:-1])==pd.Timedelta(minutes=6)):raise RuntimeError(f"cadence: {r.row_id}")
            y0,y1,x0,x1=(int(getattr(r,k)) for k in ("y0","y1","x0","x1"));tile=rates[:,y0:y1+1,x0:x1+1]
            history=tile[anchor-9:anchor+1].astype(np.float32);target=tile[anchor+1:anchor+21].astype(np.float32)
            if history.shape!=(10,128,128) or target.shape!=(20,128,128):raise RuntimeError(f"geometry: {r.row_id}")
            validity=np.isfinite(history);target_validity=np.isfinite(target)
            history_clean=np.nan_to_num(history,nan=0.,posinf=0.,neginf=0.);target_clean=np.nan_to_num(target,nan=0.,posinf=0.,neginf=0.)
            pysteps,motion,fallback=deterministic_pysteps_with_motion(history_clean)
            h=torch.from_numpy(history_clean)[None];p=torch.from_numpy(pysteps)[None];m=torch.from_numpy(motion)[None];v=torch.from_numpy(validity.astype(np.float32))[None]
            output=model(h,p,m,v);logits=output["occurrence_logits"][0].numpy();delta=output["delta_log_rate"][0].numpy();rate=output["corrected_rate"][0].numpy();prob=1/(1+np.exp(-logits))
            hnorm=(np.log1p(np.maximum(history_clean,0))-normalization["rate_log1p_mean"])/normalization["rate_log1p_std"]
            pnorm=(np.log1p(np.maximum(pysteps,0))-normalization["rate_log1p_mean"])/normalization["rate_log1p_std"]
            mnorm=(motion-np.asarray(normalization["motion_mean"])[:,None,None])/np.asarray(normalization["motion_std"])[:,None,None]
            path=tensor_dir/f"{r.row_id}.npz"
            np.savez_compressed(path,history=history_clean.astype(np.float16),history_validity=validity.astype(np.uint8),target=target_clean.astype(np.float16),target_validity=target_validity.astype(np.uint8),pysteps=pysteps.astype(np.float16),motion=motion.astype(np.float16),probability=prob.astype(np.float16),delta_log_rate=delta.astype(np.float16),corrected_rate=rate.astype(np.float16))
            rec={"system_id":system,"row_id":r.row_id,"row_type":r.row_type,"issue_time_utc":r.issue_time_utc,"source_path":str(source),"source_sha256":actual,"tensor_path":str(path),"tensor_file_sha256":sha256_file(path),"pysteps_fields_sha256":_hash_arrays(pysteps),"normalized_inputs_sha256":_hash_arrays(hnorm.astype(np.float32),pnorm.astype(np.float32),mnorm.astype(np.float32),validity.astype(np.float32)),"model_outputs_sha256":_hash_arrays(logits.astype(np.float32),delta.astype(np.float32)),"reconstructed_rate_sha256":_hash_arrays(rate.astype(np.float32)),"lk_fallback":fallback}
            records.append(rec);system_records.append(rec)
        integrity.append({"system_id":system,"source_sha256":actual,"rows":len(group),"all_geometry_valid":True,"all_cadence_valid":True})
        pd.DataFrame(records).to_csv(OUT/"generation_manifest.partial.csv",index=False)
        print(f"FINAL GENERATION {len(integrity)}/12 {system} rows={len(group)}",flush=True)
    manifest=pd.DataFrame(records).sort_values(["system_id","row_id"]);manifest.to_csv(OUT/"generation_manifest.csv",index=False)
    pd.DataFrame(integrity).to_csv(OUT/"integrity_by_system.csv",index=False)
    complete={"status":"GENERATION_COMPLETE_UNSCORED","timestamp_utc":datetime.now(timezone.utc).isoformat(),"rows":len(manifest),"systems":manifest.system_id.nunique(),"row_type_counts":rows.row_type.value_counts().to_dict(),"frozen_checks":checks,"generation_manifest_sha256":sha256_file(OUT/"generation_manifest.csv"),"all_predictions_generated_before_scoring":True,"metrics_computed":False}
    _json(OUT/"GENERATION_COMPLETE.json",complete)
    return complete


def score_final():
    marker=OUT/"GENERATION_COMPLETE.json"
    if not marker.exists():raise RuntimeError("all FINAL predictions must be sealed before scoring")
    if (OUT/"final_evaluation.json").exists():raise RuntimeError("FINAL was already scored; repeat evaluation forbidden")
    frozen_gate();manifest=pd.read_csv(OUT/"generation_manifest.csv")
    if len(manifest)!=353 or not all(sha256_file(Path(r.tensor_path))==r.tensor_file_sha256 for r in manifest.itertuples(index=False)):raise RuntimeError("generation manifest integrity failed")
    metrics=[];reliability=[];blind=[]
    for r in manifest.itertuples(index=False):
      with np.load(r.tensor_path) as z:
        truth=z["target"].astype(np.float32);valid=z["target_validity"].astype(bool);base=z["pysteps"].astype(np.float32);prob=z["probability"].astype(np.float32);rate=z["corrected_rate"].astype(np.float32);history=z["history"].astype(np.float32)
        metrics.append(row_metrics(truth,(base>.1).astype(np.float32),base,valid,.5,row_id=r.row_id,system_id=r.system_id,row_type=r.row_type,model="PySTEPS"))
        metrics.append(row_metrics(truth,prob,rate,valid,.35,row_id=r.row_id,system_id=r.system_id,row_type=r.row_type,model="ResidualV1"))
        y=((truth>.1)&valid);bins=np.minimum((prob*10).astype(int),9)
        for b in range(10):
            q=(bins==b)&valid
            if q.any():reliability.append({"row_id":r.row_id,"system_id":r.system_id,"bin":b,"count":int(q.sum()),"mean_probability":float(prob[q].mean()),"observed_frequency":float(y[q].mean())})
        def first(x):
            p=x[:-1]&x[1:];return np.where(p.any(0),(p.argmax(0)+1)*6,np.nan)
        obs=(truth>.1)&valid;pred=(prob>=.35)&valid
        eligible=(history<=.1).all(0)&~(base>.1).any(0)&np.isfinite(first(obs)) if r.row_type=="clean_initiation" else np.zeros(obs.shape[1:],bool)
        dry=~np.isfinite(first(obs));pr=np.isfinite(first(pred));blind.append({"row_id":r.row_id,"system_id":r.system_id,"eligible_locations":int(eligible.sum()),"detected_locations":int((eligible&pr).sum()),"false_area_pixels":int((dry&pr).sum()),"valid_dry_pixels":int(dry.sum())})
    met=pd.DataFrame(metrics);met.to_csv(OUT/"metrics_by_row.csv",index=False)
    event=met.groupby(["model","system_id"],as_index=False).mean(numeric_only=True);event.to_csv(OUT/"metrics_by_system.csv",index=False)
    macro=event.groupby("model",as_index=False).mean(numeric_only=True);macro.to_csv(OUT/"system_macro_metrics.csv",index=False)
    rowtype=met.groupby(["model","row_type"],as_index=False).mean(numeric_only=True);rowtype.to_csv(OUT/"row_type_metrics.csv",index=False)
    leadcols=["model"]+[f"{metric}_{minute}" for minute in (30,60,90,120) for metric in ("f1","csi","pod","far","fss18","rate_mae","brier")]
    lead=macro[leadcols];lead.to_csv(OUT/"lead_metrics.csv",index=False)
    rel=pd.DataFrame(reliability);rel.to_csv(OUT/"reliability_contributions.csv",index=False)
    relsum=rel.groupby("bin").apply(lambda x:pd.Series({"count":x["count"].sum(),"mean_probability":np.average(x.mean_probability,weights=x["count"]),"observed_frequency":np.average(x.observed_frequency,weights=x["count"])}),include_groups=False).reset_index();relsum.to_csv(OUT/"reliability_bins.csv",index=False)
    bd=pd.DataFrame(blind);bd.to_csv(OUT/"radar_blind_by_row.csv",index=False);be=bd.groupby("system_id",as_index=False).sum(numeric_only=True)
    blind_summary={"eligible_locations":int(bd.eligible_locations.sum()),"eligible_rows":int((bd.eligible_locations>0).sum()),"systems_represented":int((be.eligible_locations>0).sum()),"detection_fraction":float(bd.detected_locations.sum()/bd.eligible_locations.sum()) if bd.eligible_locations.sum() else np.nan,"false_area_fraction":float(bd.false_area_pixels.sum()/bd.valid_dry_pixels.sum()),"systems_with_any_detection":int(((be.eligible_locations>0)&(be.detected_locations>0)).sum())}
    # Same frozen diagnostic taxonomy as DEV.
    cats=[]
    for r in manifest.itertuples(index=False):
      pair=met[met.row_id.eq(r.row_id)].set_index("model")
      with np.load(r.tensor_path) as z:
        truth=z["target"].astype(np.float32);valid=z["target_validity"].astype(bool);base=z["pysteps"].astype(np.float32);obs=(truth>.1)&valid;bp=(base>.1)&valid;ow=obs.mean();bw=bp.mean()
        if r.row_type=="clean_initiation" and obs.any() and not bp.any():cat="no initiation signal"
        elif ow>bw+.02:cat="growth underprediction"
        elif bw>ow+.02:cat="decay persistence"
        elif float(pair.loc["PySTEPS","fss18"])>float(pair.loc["PySTEPS","f1"])+.12:cat="displacement"
        else:cat="intensity error"
      for model in ("PySTEPS","ResidualV1"):cats.append({"row_id":r.row_id,"system_id":r.system_id,"category":cat,"model":model,"f1":pair.loc[model,"f1"],"fss18":pair.loc[model,"fss18"],"rate_mae":pair.loc[model,"rate_mae"]})
    cats=pd.DataFrame(cats);cats.to_csv(OUT/"failure_category_by_row.csv",index=False)
    catsum=cats.groupby(["category","model"],as_index=False).agg(cases=("row_id","nunique"),f1=("f1","mean"),fss18=("fss18","mean"),rate_mae=("rate_mae","mean"));catsum.to_csv(OUT/"failure_category_metrics.csv",index=False)
    hard=met[met.row_type.eq("hard_negative")].groupby("model").agg(mean_probability=("mean_probability","mean"),maximum_probability=("max_probability","max"),mean_wet_area=("wet_area","mean"),maximum_wet_area=("wet_area","max"),false_initiation_fraction=("false_initiation","mean"),brier=("brier","mean")).reset_index();hard.to_csv(OUT/"hard_negative_metrics.csv",index=False)
    a=event[event.model.eq("ResidualV1")].set_index("system_id");b=event[event.model.eq("PySTEPS")].set_index("system_id");delta=a.f1-b.f1;am=macro.set_index("model").loc["ResidualV1"];bm=macro.set_index("model").loc["PySTEPS"]
    # Holistic predeclared interpretation: principal categorical/spatial/probability finding plus broad event replication.
    categorical=(am.f1>bm.f1 and am.fss18>bm.fss18);probability=am.brier<bm.brier;consistent=int((delta>0).sum())>=7;dry=float(hard.set_index("model").loc["ResidualV1","mean_wet_area"])<=.01
    if categorical and probability and consistent and dry:classification="FINAL VALIDATED — RESIDUAL V1 GENERALIZES"
    elif (am.f1>bm.f1 or am.fss18>bm.fss18 or probability):classification="FINAL MIXED — RESIDUAL BENEFIT PARTIALLY GENERALIZES"
    else:classification="FINAL NEGATIVE — RESIDUAL BENEFIT FAILS TO GENERALIZE"
    result={"status":"CONSUMED_FINAL","evaluation_timestamp_utc":datetime.now(timezone.utc).isoformat(),"final_manifest_sha256":EXPECTED["final_manifest"],"checkpoint_sha256":EXPECTED["checkpoint"],"normalization_sha256":EXPECTED["normalization"],"configuration_sha256":EXPECTED["configuration"],"model_source_sha256":EXPECTED["model_source"],"operating_threshold":.35,"systems":12,"rows":353,"system_macro_metrics":macro.set_index("model").to_dict("index"),"system_metrics":event.to_dict("records"),"lead_metrics":lead.to_dict("records"),"row_type_metrics":rowtype.to_dict("records"),"hard_negative_metrics":hard.to_dict("records"),"failure_category_metrics":catsum.to_dict("records"),"probability_metrics":{"reliability_bins":relsum.to_dict("records"),"raw_uncalibrated":True},"initiation_diagnostic":blind_summary,"systems_strict_f1_improvement":int((delta>0).sum()),"systems_tied_within_0.005":int((delta.abs()<=.005).sum()),"systems_worse_more_than_0.005":int((delta<-.005).sum()),"median_f1_delta":float(delta.median()),"median_fss18_delta":float((a.fss18-b.fss18).median()),"classification":classification,"post_final_retuning_permitted":False}
    _json(OUT/"final_evaluation.json",result)
    return result


def write_hash_manifest():
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="hash_manifest.json")
    value={"status":"CONSUMED_FINAL","generated_utc":datetime.now(timezone.utc).isoformat(),"files":[{"path":str(p),"sha256":sha256_file(p),"bytes":p.stat().st_size} for p in files]}
    _json(OUT/"hash_manifest.json",value);return value
