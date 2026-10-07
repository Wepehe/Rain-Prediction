"""Stage 11 diagnostic-only GOES C13 precursor analysis on consumed Stage 8 development."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.request
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import rankdata, spearmanr

from ..data.manifest import sha256_file
from ..data.s3 import list_public_bucket, object_url
from ..preprocessing.grid import transform_coordinates
from ..sample_evaluation import _downsample_max

ROOT = Path("artifacts/stage_11")
SOURCE = ROOT / "goes_c13_sources"
COLD_K = 235.0
RAPID_K_PER_HOUR = 8.0
LATENCY_MINUTES = 10
COOLING_MINUTES = (10, 20, 30, 60)
GOES_FEATURES = [
    "c13_mean_k", "c13_min_k", "c13_p10_k", "cold_cloud_fraction",
    "cooling_10_mean_k", "cooling_20_mean_k", "cooling_30_mean_k", "cooling_60_mean_k",
    "cooling_30_p90_k", "rapid_cooling_fraction",
    "radar_uncovered_cold_fraction", "pysteps_uncovered_cold_fraction",
    "radar_uncovered_rapid_fraction", "pysteps_uncovered_rapid_fraction",
]
CONTEXT = ["lead_fraction", "current_radar_wet_fraction", "pysteps_wet_fraction"]
CURRENT = ["c13_mean_k", "c13_min_k", "c13_p10_k", "cold_cloud_fraction",
           "radar_uncovered_cold_fraction", "pysteps_uncovered_cold_fraction"]
COOLING = [x for x in GOES_FEATURES if x not in CURRENT]
START_RE = re.compile(r"_s(\d{13})")
END_RE = re.compile(r"_e(\d{13})")
CREATE_RE = re.compile(r"_c(\d{13,14})")


@lru_cache(maxsize=24)
def _read_goes(path_text: str):
    import xarray as xr
    from pyproj import CRS
    with xr.open_dataset(path_text, engine="h5netcdf") as dataset:
        projection=dataset.goes_imager_projection.attrs; height=float(projection["perspective_point_height"])
        geos=CRS.from_proj4("+proj=geos "f"+h={height} +lon_0={float(projection['longitude_of_projection_origin'])} "f"+a={float(projection['semi_major_axis'])} +b={float(projection['semi_minor_axis'])} "f"+sweep={projection['sweep_angle_axis']} +units=m +no_defs")
        x=dataset.x.values*height; y=dataset.y.values*height; values=dataset.CMI.values.astype(np.float32)
    if y[0]>y[-1]: y=y[::-1]; values=values[::-1]
    return x,y,values,geos.to_string()


def _interpolate_c13(path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    x,y,values,crs=_read_goes(path.as_posix()); tx,ty=transform_coordinates(lon,lat,source_crs="EPSG:4326",destination_crs=crs)
    interp=RegularGridInterpolator((y,x),values,bounds_error=False,fill_value=np.nan)
    return interp(np.column_stack((ty.ravel(),tx.ravel()))).reshape(lon.shape).astype(np.float32)


def _file_time(key: str, regex: re.Pattern[str]) -> pd.Timestamp:
    value = regex.search(key).group(1)  # type: ignore[union-attr]
    return pd.Timestamp(datetime.strptime(value[:13], "%Y%j%H%M%S").replace(tzinfo=UTC))


def _event_path(event: str, source: object = None) -> Path:
    options = ([Path(str(source))] if isinstance(source, str) and source else []) + [
        Path(f"data/stage8_qualification/{event}/processed/events/{event}.npz"),
        Path(f"data/s8/e01/processed/events/{event}.npz"),
        Path(f"data/s8/e02/processed/events/{event}.npz"),
    ]
    return next(path for path in options if path.exists())


def _rows() -> pd.DataFrame:
    pos = pd.read_csv("artifacts/stage_8/qualification/stage8_development_object_manifest.csv")
    pos = pos[pos.split.eq("development")]
    neg = pd.read_csv("artifacts/stage_8/qualification/stage8_hard_negative_manifest.csv")
    neg = neg[neg.split.eq("development")]
    a = pos.rename(columns={"object_id": "row_id"}).assign(role="positive_initiation")
    b = neg.rename(columns={"hard_negative_id": "row_id"}).assign(role="hard_negative")
    columns = ["row_id", "role", "event_id", "independent_system_group", "issue_time_utc",
               "tile_y_min", "tile_y_max", "tile_x_min", "tile_x_max"]
    a = a[columns + ["radar_source_path"]]
    b["radar_source_path"] = None
    return pd.concat([a, b[columns + ["radar_source_path"]]], ignore_index=True)


def _requirements(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for issue_text in sorted(rows.issue_time_utc.unique()):
        issue = pd.Timestamp(issue_text)
        if issue.tzinfo is None: issue = issue.tz_localize("UTC")
        else: issue = issue.tz_convert("UTC")
        current = issue - pd.Timedelta(minutes=LATENCY_MINUTES)
        for lag in (0, *COOLING_MINUTES):
            records.append({"issue_time_utc": issue.isoformat(), "role": "current" if lag == 0 else f"cooling_{lag}_baseline",
                            "requested_time_utc": (current-pd.Timedelta(minutes=lag)).isoformat()})
    return pd.DataFrame(records)


def audit_and_download() -> None:
    ROOT.mkdir(parents=True, exist_ok=True); SOURCE.mkdir(parents=True, exist_ok=True)
    req = _requirements(_rows())
    requested = pd.to_datetime(req.requested_time_utc, utc=True)
    hours = sorted(set(t.floor("h") for t in requested))
    objects = {}
    for number, hour in enumerate(hours, 1):
        prefix = f"ABI-L2-CMIPC/{hour:%Y}/{hour:%j}/{hour:%H}/"
        found = [o for o in list_public_bucket("noaa-goes16", prefix) if "C13_G16" in o.key]
        if not found: raise RuntimeError(f"no GOES-16 C13 objects under {prefix}")
        objects[hour] = found
        if number % 20 == 0: print(f"[stage11-audit] archive hours {number}/{len(hours)}", flush=True)
    selected = []
    for r in req.itertuples(index=False):
        issue = pd.Timestamp(r.issue_time_utc); wanted = pd.Timestamp(r.requested_time_utc)
        candidates = []
        for h in (wanted.floor("h")-pd.Timedelta(hours=1), wanted.floor("h"), wanted.floor("h")+pd.Timedelta(hours=1)):
            if h not in objects:
                prefix = f"ABI-L2-CMIPC/{h:%Y}/{h:%j}/{h:%H}/"
                objects[h] = [o for o in list_public_bucket("noaa-goes16", prefix) if "C13_G16" in o.key]
            candidates.extend(objects[h])
        causal = [o for o in candidates if _file_time(o.key, END_RE) <= issue and _file_time(o.key, CREATE_RE) <= issue]
        if not causal: raise RuntimeError(f"no causal scan for {r.issue_time_utc} / {r.requested_time_utc}")
        obj = min(causal, key=lambda o: abs(_file_time(o.key, START_RE)-wanted))
        path = SOURCE / Path(obj.key).name
        selected.append({"issue_time_utc": r.issue_time_utc, "source_role": r.role,
                         "requested_time_utc": r.requested_time_utc, "satellite": "GOES-16",
                         "product": "ABI-L2-CMIPC", "channel": "C13", "bucket": "noaa-goes16",
                         "key": obj.key, "scan_start_utc": _file_time(obj.key, START_RE).isoformat(),
                         "scan_end_utc": _file_time(obj.key, END_RE).isoformat(),
                         "archive_creation_utc": _file_time(obj.key, CREATE_RE).isoformat(),
                         "local_path": path.as_posix(), "etag": obj.etag})
    table = pd.DataFrame(selected)
    unique = table.drop_duplicates("key").to_dict("records")
    def download(r):
        path = Path(r["local_path"])
        if not path.exists():
            for attempt in range(5):
                try:
                    data = urllib.request.urlopen(object_url(r["bucket"], r["key"]), timeout=180).read()
                    path.write_bytes(data); break
                except Exception:
                    if attempt == 4: raise
                    time.sleep(2**attempt)
        return r["key"], path.stat().st_size, sha256_file(path)
    metadata = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for n, result in enumerate(ex.map(download, unique), 1):
            metadata[result[0]] = result[1:]
            if n % 50 == 0: print(f"[stage11-audit] files {n}/{len(unique)}", flush=True)
    table["bytes"] = [metadata[k][0] for k in table.key]
    table["sha256"] = [metadata[k][1] for k in table.key]
    table["causal"] = (pd.to_datetime(table.scan_end_utc, utc=True) <= pd.to_datetime(table.issue_time_utc, utc=True)) & (pd.to_datetime(table.archive_creation_utc, utc=True) <= pd.to_datetime(table.issue_time_utc, utc=True))
    table["actual_separation_minutes"] = table.groupby("issue_time_utc").scan_start_utc.transform(lambda s: (pd.to_datetime(s[s.index[0]], utc=True)-pd.to_datetime(s, utc=True)).dt.total_seconds()/60)
    table.to_csv(ROOT/"goes_source_audit.csv", index=False)
    summary = {"satellite":"GOES-16", "product":"ABI-L2-CMIPC", "channel":"C13",
               "development_issue_times":int(table.issue_time_utc.nunique()), "required_records":len(table),
               "unique_files":len(unique), "available_records":int(table.local_path.map(lambda p: Path(p).exists()).sum()),
               "all_causal":bool(table.causal.all()), "years":sorted(pd.to_datetime(table.issue_time_utc,utc=True).dt.year.unique().tolist()),
               "latency_minutes":LATENCY_MINUTES, "sealed_final_accessed":False, "final_features_generated":False}
    (ROOT/"goes_availability_summary.json").write_text(json.dumps(summary,indent=2))


def materialize_descriptors() -> None:
    rows = _rows(); sources = pd.read_csv(ROOT/"goes_source_audit.csv")
    source_map = {(r.issue_time_utc,r.source_role):r for r in sources.itertuples(index=False)}
    cache_manifest = pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv").set_index("row_id")
    event_cache = {}; records=[]
    for number, r in enumerate(rows.itertuples(index=False), 1):
        if r.event_id not in event_cache:
            z=np.load(_event_path(r.event_id,r.radar_source_path)); event_cache[r.event_id]=(_downsample_max(z["rate_mm_hr"],2),pd.to_datetime(z["times"],utc=True),z["latitude"],z["longitude"])
        rates,times,lat,lon=event_cache[r.event_id]; y0,y1,x0,x1=map(int,[r.tile_y_min,r.tile_y_max,r.tile_x_min,r.tile_x_max])
        lat=lat[:rates.shape[1]*2:2] if len(lat)>rates.shape[1] else lat; lon=lon[:rates.shape[2]*2:2] if len(lon)>rates.shape[2] else lon
        glon,glat=np.meshgrid(lon[x0:x1+1],lat[y0:y1+1]); issue=pd.Timestamp(r.issue_time_utc)
        current_rec=source_map[(issue.isoformat(),"current")]; current=_interpolate_c13(Path(current_rec.local_path),glon,glat)
        cool={}
        for window in COOLING_MINUTES:
            base=source_map[(issue.isoformat(),f"cooling_{window}_baseline")]
            cool[window]=_interpolate_c13(Path(base.local_path),glon,glat)-current
        rapid=cool[30]>=(RAPID_K_PER_HOUR*.5); cold=current<COLD_K
        ix=times.get_indexer([issue])[0]; radar=rates[ix,y0:y1+1,x0:x1+1]; z=np.load(cache_manifest.loc[r.row_id,"cache_path"]); py=z["pysteps_rate"].astype(float)
        base={"row_id":r.row_id,"role":r.role,"event_id":r.event_id,"independent_system_group":r.independent_system_group,"issue_time_utc":issue.isoformat(),
              "c13_mean_k":np.nanmean(current),"c13_min_k":np.nanmin(current),"c13_p10_k":np.nanpercentile(current,10),"cold_cloud_fraction":np.nanmean(cold),
              **{f"cooling_{w}_mean_k":np.nanmean(cool[w]) for w in COOLING_MINUTES},"cooling_30_p90_k":np.nanpercentile(cool[30],90),"rapid_cooling_fraction":np.nanmean(rapid),
              "radar_uncovered_cold_fraction":np.nanmean(cold&(radar<.1)),"radar_uncovered_rapid_fraction":np.nanmean(rapid&(radar<.1)),
              "goes_valid_fraction":min([np.isfinite(current).mean(), *[np.isfinite(cool[w]).mean() for w in COOLING_MINUTES]])}
        for j in range(20):
            pywet=py[j]>=.1
            records.append({**base,"lead_minutes":6*(j+1),"lead_fraction":(j+1)/20,"current_radar_wet_fraction":np.nanmean(radar>=.1),"pysteps_wet_fraction":np.nanmean(pywet),
                            "pysteps_uncovered_cold_fraction":np.nanmean(cold&~pywet),"pysteps_uncovered_rapid_fraction":np.nanmean(rapid&~pywet)})
        if number%25==0: print(f"[stage11-descriptors] rows {number}/{len(rows)}",flush=True)
    out=pd.DataFrame(records); out.to_csv(ROOT/"causal_goes_descriptors_by_row_lead.csv",index=False)
    pd.DataFrame([{"cold_cloud_threshold_k":COLD_K,"rapid_cooling_threshold_k_per_hour":RAPID_K_PER_HOUR,"cooling_sign":"baseline minus current; positive is cooling","remapping":"frozen scipy linear griddata used by Stage 4B","availability_latency_minutes":LATENCY_MINUTES,"frozen_before_truth_scoring":True}]).to_csv(ROOT/"descriptor_predeclaration.csv",index=False)


def _auc(y,p):
    y=np.asarray(y,int); n1=y.sum(); n0=len(y)-n1
    return np.nan if not n1 or not n0 else float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def _fit(x,y,groups):
    mean=x.mean(0); std=x.std(0); std[std==0]=1; z=(x-mean)/std; design=np.c_[np.ones(len(z)),z]
    counts=pd.Series(groups).value_counts(); w=np.array([1/counts[g] for g in groups]); w/=w.sum()
    def fg(beta):
        p=1/(1+np.exp(-np.clip(design@beta,-30,30))); loss=-np.sum(w*(y*np.log(np.clip(p,1e-8,1))+(1-y)*np.log(np.clip(1-p,1e-8,1))))+np.sum(beta[1:]**2)
        grad=design.T@(w*(p-y)); grad[1:]+=2*beta[1:]; return loss,grad
    res=minimize(lambda b:fg(b),np.zeros(design.shape[1]),jac=True,method="L-BFGS-B")
    return mean,std,res.x,bool(res.success)


def _predict(x,mean,std,beta): return 1/(1+np.exp(-np.clip(np.c_[np.ones(len(x)),(x-mean)/std]@beta,-30,30)))


def _metric(y,p):
    q=p>=.5; tp=np.sum(q&(y==1)); fp=np.sum(q&(y==0)); fn=np.sum(~q&(y==1))
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
    return {"auc":_auc(y,p),"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.,"brier":np.mean((p-y)**2)}


def evaluate() -> None:
    d=pd.read_csv(ROOT/"causal_goes_descriptors_by_row_lead.csv")
    oracle=pd.read_csv("artifacts/stage_9/source_advantage/oracle_source_advantage_by_row_lead.csv")
    oracle=oracle[oracle.collection.eq("stage8_nested_development")][["row_id","lead_minutes","pysteps_failure_category"]]
    d=d.merge(oracle,on=["row_id","lead_minutes"],how="left",validate="one_to_one")
    d["failure_target"]=d.pysteps_failure_category.isin(["no_initiation_signal","growth_underprediction"]).astype(int)
    availability=d.drop_duplicates("row_id")[["row_id","role","event_id","independent_system_group","issue_time_utc","goes_valid_fraction"]]
    availability.to_csv(ROOT/"tile_availability.csv",index=False)
    positive=d[d.role.eq("positive_initiation")&d.goes_valid_fraction.gt(0)].copy().reset_index(drop=True); groups=positive.independent_system_group.to_numpy(); rng=np.random.default_rng(1101)
    variants={"context_only":CONTEXT,"context_plus_current_c13":CONTEXT+CURRENT,"context_plus_cooling":CONTEXT+COOLING,"context_plus_all_goes":CONTEXT+GOES_FEATURES}
    predictions=[]; fits=[]
    for name,features in variants.items():
        x=positive[features].to_numpy(float); pred=np.zeros(len(positive)); zero=np.zeros(len(positive)); shuffled_global=np.zeros(len(positive)); shuffled_event=np.zeros(len(positive))
        global_perm=rng.permutation(len(positive))
        for hold in sorted(np.unique(groups)):
            tr=groups!=hold; te=~tr; mean,std,beta,success=_fit(x[tr],positive.failure_target.to_numpy()[tr],groups[tr]); pred[te]=_predict(x[te],mean,std,beta)
            destroyed=x[te].copy(); event_shuffle=x[te].copy(); global_shuffle=x[te].copy()
            if name!="context_only":
                gi=[features.index(f) for f in features if f in GOES_FEATURES]; destroyed[:,gi]=mean[gi]; event_shuffle[:,gi]=event_shuffle[rng.permutation(te.sum())][:,gi]; global_shuffle[:,gi]=x[global_perm[:te.sum()]][:,gi]
            zero[te]=_predict(destroyed,mean,std,beta); shuffled_event[te]=_predict(event_shuffle,mean,std,beta); shuffled_global[te]=_predict(global_shuffle,mean,std,beta)
            fits.append({"model":name,"held_out_system":hold,"success":success,"training_systems":"|".join(sorted(set(groups[tr]))),"normalization_training_only":True,"intercept":beta[0],**{f"coef_{f}":v for f,v in zip(features,beta[1:])}})
        for mode,p in [("normal",pred),("goes_removed",zero),("goes_shuffled_global",shuffled_global),("goes_shuffled_within_event",shuffled_event)]:
            for i,value in enumerate(p): predictions.append({"model":name,"destruction":mode,"row_index":i,"probability":value})
    pred=pd.DataFrame(predictions); key=positive.reset_index().rename(columns={"index":"row_index"})
    pred=pred.merge(key[["row_index","row_id","independent_system_group","lead_minutes","failure_target","pysteps_failure_category"]],on="row_index",validate="many_to_one")
    pred.to_csv(ROOT/"loeo_predictions.csv",index=False); pd.DataFrame(fits).to_csv(ROOT/"loeo_fits.csv",index=False)
    metrics=[]
    for (model,destruction,event),t in pred.groupby(["model","destruction","independent_system_group"]): metrics.append({"model":model,"destruction":destruction,"event":event,"samples":len(t),"prevalence":t.failure_target.mean(),**_metric(t.failure_target.to_numpy(),t.probability.to_numpy())})
    em=pd.DataFrame(metrics); em.to_csv(ROOT/"loeo_metrics_by_event.csv",index=False); em.groupby(["model","destruction"],as_index=False).mean(numeric_only=True).to_csv(ROOT/"loeo_metrics_event_macro.csv",index=False)
    lead_records=[]
    normal_pred=pred[pred.destruction.eq("normal")].copy(); normal_pred["lead_group"]=pd.cut(normal_pred.lead_minutes,[0,30,60,90,120],labels=["0-30","30-60","60-90","90-120"])
    for (model,lead,event),t in normal_pred.groupby(["model","lead_group","independent_system_group"],observed=True): lead_records.append({"model":model,"lead_group":lead,"event":event,**_metric(t.failure_target.to_numpy(),t.probability.to_numpy())})
    lead_table=pd.DataFrame(lead_records); lead_table.groupby(["model","lead_group"],observed=True,as_index=False).mean(numeric_only=True).to_csv(ROOT/"loeo_metrics_event_macro_by_lead.csv",index=False)
    # Training-event base-rate and majority nulls are genuinely outer-event predictions.
    null=[]
    for hold in sorted(np.unique(groups)):
        tr=groups!=hold; te=~tr; rate=positive.failure_target.to_numpy()[tr].mean()
        for name,p in [("training_event_base_rate",np.full(te.sum(),rate)),("training_majority",np.full(te.sum(),float(rate>=.5)))]: null.append({"event":hold,"model":name,**_metric(positive.failure_target.to_numpy()[te],p)})
    pd.DataFrame(null).to_csv(ROOT/"null_comparators_by_event.csv",index=False)
    # Descriptor distributions, lead dependence, sign stability, and hard negatives.
    positive.groupby(["pysteps_failure_category"],as_index=False)[GOES_FEATURES].mean().to_csv(ROOT/"descriptors_by_failure_category.csv",index=False)
    positive.assign(lead_group=pd.cut(positive.lead_minutes,[0,30,60,90,120],labels=["0-30","30-60","60-90","90-120"])).groupby(["lead_group","failure_target"],observed=True,as_index=False)[GOES_FEATURES].mean().to_csv(ROOT/"descriptors_by_lead_and_failure.csv",index=False)
    assoc=[]
    for feature in GOES_FEATURES:
        signs=[]
        for event,t in positive.groupby("independent_system_group"):
            rho=spearmanr(t[feature],t.failure_target).statistic; signs.append(rho); assoc.append({"feature":feature,"event":event,"spearman":rho})
        assoc.append({"feature":feature,"event":"EVENT_CONSISTENCY","spearman":np.nan,"positive_systems":int(np.sum(np.asarray(signs)>0)),"negative_systems":int(np.sum(np.asarray(signs)<0)),"negligible_systems":int(np.sum(np.abs(np.nan_to_num(signs))<.05))})
    pd.DataFrame(assoc).to_csv(ROOT/"descriptor_event_consistency.csv",index=False)
    d[d.role.eq("hard_negative")&d.goes_valid_fraction.gt(0)].groupby("independent_system_group",as_index=False)[["cold_cloud_fraction","rapid_cooling_fraction","cooling_30_p90_k","radar_uncovered_cold_fraction"]].mean().to_csv(ROOT/"hard_negative_descriptors_by_event.csv",index=False)
    # Calibration and category discrimination for the full frozen diagnostic.
    full=pred[(pred.model=="context_plus_all_goes")&(pred.destruction=="normal")].copy(); full["bin"]=pd.cut(full.probability,np.linspace(0,1,11),include_lowest=True)
    full.groupby("bin",observed=True).agg(samples=("failure_target","size"),mean_probability=("probability","mean"),observed_frequency=("failure_target","mean")).reset_index().to_csv(ROOT/"loeo_calibration.csv",index=False)
    cats=[]; context_pred=pred[(pred.model=="context_only")&(pred.destruction=="normal")]
    for cat,t in full.groupby("pysteps_failure_category"):
        row={"category":cat,"samples":len(t),"events":t.independent_system_group.nunique(),"mean_diagnostic_probability":t.probability.mean()}; auc_all=[]; auc_context=[]
        for event,tt in full.groupby("independent_system_group"):
            yy=tt.pysteps_failure_category.eq(cat).astype(int).to_numpy(); auc_all.append(_auc(yy,tt.probability.to_numpy())); cp=context_pred[context_pred.independent_system_group.eq(event)]; auc_context.append(_auc(yy,cp.probability.to_numpy()))
        row.update(event_macro_one_vs_rest_auc=np.nanmean(auc_all),context_event_macro_one_vs_rest_auc=np.nanmean(auc_context)); cats.append(row)
    pd.DataFrame(cats).to_csv(ROOT/"failure_category_discrimination.csv",index=False)
    macro=pd.read_csv(ROOT/"loeo_metrics_event_macro.csv"); normal=macro[macro.destruction.eq("normal")].set_index("model"); context=normal.loc["context_only"]; both=normal.loc["context_plus_all_goes"]
    per=em[em.destruction.eq("normal")].pivot(index="event",columns="model",values="auc"); gains=per.context_plus_all_goes-per.context_only
    classification="PROMISING" if both.auc>=context.auc+.03 and (gains>.01).sum()>=4 else ("WEAK / REGIME-DEPENDENT" if both.auc>context.auc and (gains>0).sum()>=2 else "NOT USEFUL")
    decision={"classification":classification,"context_event_macro_auc":context.auc,"context_plus_goes_event_macro_auc":both.auc,"auc_increment":both.auc-context.auc,"systems_auc_improved":int((gains>0).sum()),"systems_auc_improved_gt_0_01":int((gains>.01).sum()),"correction_experiment_justified":classification=="PROMISING","promotable_model_trained":False,"sealed_final_accessed":False}
    (ROOT/"stage11_decision.json").write_text(json.dumps(decision,indent=2))
    outputs={p.name:sha256_file(p) for p in ROOT.glob("*.csv")}
    manifest={"status":"DIAGNOSTIC COMPLETE","rows":int(d.row_id.nunique()),"positive_rows":int(d[d.role.eq('positive_initiation')].row_id.nunique()),"hard_negative_rows":int(d[d.role.eq('hard_negative')].row_id.nunique()),"usable_positive_rows":int(positive.row_id.nunique()),"unavailable_tile_rows":int(availability.goes_valid_fraction.eq(0).sum()),"systems":int(positive.independent_system_group.nunique()),"promotable_model_trained":False,"sealed_final_accessed":False,"final_goes_materialized":False,"source_summary_sha256":sha256_file(ROOT/"goes_availability_summary.json"),"outputs":outputs,"decision_sha256":sha256_file(ROOT/"stage11_decision.json")}
    (ROOT/"stage11_manifest.json").write_text(json.dumps(manifest,indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=["audit","materialize","evaluate","all"],default="all",nargs="?"); args=parser.parse_args()
    if args.phase in ("audit","all"): audit_and_download()
    if args.phase in ("materialize","all"): materialize_descriptors()
    if args.phase in ("evaluate","all"): evaluate()


if __name__=="__main__": main()
