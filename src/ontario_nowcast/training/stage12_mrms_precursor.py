"""Stage 12 diagnostic-only richer-MRMS precursor analysis."""
from __future__ import annotations
import argparse, concurrent.futures, gzip, json, re, tempfile, time, urllib.request
from datetime import UTC, datetime
from pathlib import Path
import eccodes, numpy as np, pandas as pd, requests
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr
from ..data.manifest import sha256_file
from ..data.s3 import list_public_bucket, object_url
from ..sample_evaluation import _downsample_max

ROOT=Path("artifacts/stage_12"); SOURCE=ROOT/"mrms_sources"; BUCKET="noaa-mrms-pds"
PRODUCTS={
 "merged_reflectivity":{"archive":"MergedReflectivityQCComposite_00.50","units":"dBZ","missing":"-99; no coverage -999"},
 "lowest_reflectivity":{"archive":"ReflectivityAtLowestAltitude_00.50","units":"dBZ","missing":"-99; no coverage -999"},
 "echo_top_30":{"archive":"EchoTop_30_00.50","units":"km MSL","missing":"-1; no coverage -3"},
 "echo_top_50":{"archive":"EchoTop_50_00.50","units":"km MSL","missing":"-1; no coverage -3"},
 "vil":{"archive":"VIL_00.50","units":"kg m-2","missing":"-1; no coverage -3"},
}
AVAILABILITY=.90; MAX_AGE_MIN=4.; REFLECTIVITY_DBZ=20.; REFLECTIVITY_GROWTH_DBZ=5.; ECHOTOP_GROWTH_KM=1.; VIL_GROWTH=2.
CONTEXT=["lead_fraction","current_precip_wet_fraction","pysteps_wet_fraction"]
GROUPS={"reflectivity":["merged_cov20","merged_p90","merged_max","merged_growth_cov","merged_growth_p90","lowest_cov20","lowest_p90","lowest_max","lowest_growth_cov","lowest_growth_p90","reflectivity_only_fraction"],
 "echo_top":["et30_cov","et30_p90","et30_max","et30_growth_cov","et30_growth_p90","et50_cov","et50_p90","et50_max","et50_growth_cov","et50_growth_p90"],
 "vil":["vil_cov","vil_p90","vil_growth_cov","vil_growth_p90"],
 "mask":["elevated_echo_without_surface_rain_fraction"]}
RADAR_FEATURES=sum(GROUPS.values(),[]); STAMP=re.compile(r"_(\d{8}-\d{6})\.grib2")

def stamp(key): return pd.Timestamp(datetime.strptime(STAMP.search(key).group(1),"%Y%m%d-%H%M%S").replace(tzinfo=UTC))
def event_path(event,source=None):
 opts=([Path(str(source))] if isinstance(source,str) and source else [])+[Path(f"data/stage8_qualification/{event}/processed/events/{event}.npz"),Path(f"data/s8/e01/processed/events/{event}.npz"),Path(f"data/s8/e02/processed/events/{event}.npz")]
 return next(p for p in opts if p.exists())
def rows():
 p=pd.read_csv("artifacts/stage_8/qualification/stage8_development_object_manifest.csv");p=p[p.split.eq("development")].rename(columns={"object_id":"row_id"}).assign(role="positive_initiation")
 n=pd.read_csv("artifacts/stage_8/qualification/stage8_hard_negative_manifest.csv");n=n[n.split.eq("development")].rename(columns={"hard_negative_id":"row_id"}).assign(role="hard_negative",radar_source_path=None)
 cols=["row_id","role","event_id","independent_system_group","issue_time_utc","tile_y_min","tile_y_max","tile_x_min","tile_x_max","radar_source_path"]
 return pd.concat([p[cols],n[cols]],ignore_index=True)

def audit():
 ROOT.mkdir(parents=True,exist_ok=True); r=rows(); issues=pd.to_datetime(r.issue_time_utc,utc=True); days=sorted(set((t-pd.Timedelta(minutes=35)).floor("d") for t in issues)|set(t.floor("d") for t in issues)); listings={}
 for product,meta in PRODUCTS.items():
  for day in days: listings[(product,day)]=[(stamp(o.key),o) for o in list_public_bucket(BUCKET,f"CONUS/{meta['archive']}/{day:%Y%m%d}/")]
 records=[]
 for rr in r.itertuples(index=False):
  issue=pd.Timestamp(rr.issue_time_utc)
  for product,meta in PRODUCTS.items():
   for role,wanted in [("current",issue),("baseline_30min",issue-pd.Timedelta(minutes=30))]:
    candidates=[]
    for day in {wanted.floor("d"),wanted.floor("d")-pd.Timedelta(days=1)}: candidates+=listings.get((product,day),[])
    causal=[x for x in candidates if x[0]<=wanted]; selected=max(causal,key=lambda x:x[0]) if causal else None; obj=selected[1] if selected else None; source_time=selected[0] if selected else None; age=(wanted-source_time).total_seconds()/60 if obj else np.nan; available=bool(obj and age<=MAX_AGE_MIN)
    records.append({"row_id":rr.row_id,"role":rr.role,"event_id":rr.event_id,"independent_system_group":rr.independent_system_group,"issue_time_utc":issue.isoformat(),"product":product,"archive_product":meta["archive"],"source_role":role,"requested_time_utc":wanted.isoformat(),"source_time_utc":source_time.isoformat() if obj else None,"age_minutes":age,"available":available,"causal":bool(obj and source_time<=issue),"key":obj.key if obj else None,"etag":obj.etag if obj else None,"bytes":obj.size if obj else None})
 a=pd.DataFrame(records); coverage=a.groupby(["product","independent_system_group"],as_index=False).available.mean().rename(columns={"available":"availability_fraction"}); retained=coverage.groupby("product").availability_fraction.min().ge(AVAILABILITY); coverage["product_retained"]=coverage["product"].map(retained); coverage.to_csv(ROOT/"product_coverage_by_system.csv",index=False);a["product_retained"]=a["product"].map(retained);a.to_csv(ROOT/"source_availability_audit.csv",index=False)
 definitions=[]
 for p,m in PRODUCTS.items(): definitions.append({"product":p,"archive_product":m["archive"],"units":m["units"],"spatial_resolution":"0.01 degree (~1 km)","nominal_cadence":"2 min","timestamp_semantics":"filename observation/valid time; latest frame not after requested issue/history time","missing_codes":m["missing"],"first_archive_date":"2020-10 (public operational archive)","last_archive_date":"ongoing; audited through 2022 development periods","availability_gate":f">={AVAILABILITY:.0%} in every retained system","retained":bool(retained[p])})
 pd.DataFrame(definitions).to_csv(ROOT/"product_definitions.csv",index=False)
 summary={"criterion":f">={AVAILABILITY:.0%} expected current/baseline records in every system","retained_products":[p for p,v in retained.items() if v],"rejected_products":[p for p,v in retained.items() if not v],"rows":len(r),"systems":r.independent_system_group.nunique(),"all_selected_causal":bool(a[a.available].causal.all()),"sealed_final_accessed":False}
 (ROOT/"availability_gate.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

def download():
 SOURCE.mkdir(parents=True,exist_ok=True); a=pd.read_csv(ROOT/"source_availability_audit.csv"); a=a[a.available&a.product_retained].copy(); unique=a.drop_duplicates("key").copy(); unique["local_path"]=[(SOURCE/Path(k).name).as_posix() for k in unique.key]
 def one(r):
  path=Path(r.local_path)
  if not path.exists():
   for attempt in range(5):
    try:path.write_bytes(urllib.request.urlopen(object_url(BUCKET,r.key),timeout=180).read());break
    except Exception:
     if attempt==4:raise
     time.sleep(2**attempt)
  return r.key,path.as_posix(),path.stat().st_size,sha256_file(path)
 got={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
  for i,v in enumerate(ex.map(one,unique.itertuples(index=False)),1):got[v[0]]=v[1:];print(f"[stage12-download] {i}/{len(unique)}",flush=True) if i%100==0 else None
 a["local_path"]=[got[k][0] for k in a.key];a["downloaded_bytes"]=[got[k][1] for k in a.key];a["sha256"]=[got[k][2] for k in a.key];a.to_csv(ROOT/"source_manifest.csv",index=False)
 (ROOT/"download_summary.json").write_text(json.dumps({"selected_records":len(a),"unique_files":len(unique),"checksums_complete":bool(a.sha256.str.len().eq(64).all()),"sealed_final_accessed":False},indent=2))

def decode_target(path,product,tlat,tlon):
 message=eccodes.codes_new_from_message(gzip.open(path,"rb").read())
 try:
  ni=int(eccodes.codes_get(message,"Ni"));lat0=float(eccodes.codes_get(message,"latitudeOfFirstGridPointInDegrees"));lon0=((float(eccodes.codes_get(message,"longitudeOfFirstGridPointInDegrees"))+180)%360)-180;di=float(eccodes.codes_get(message,"iDirectionIncrementInDegrees"));dj=float(eccodes.codes_get(message,"jDirectionIncrementInDegrees"));iy=np.rint((lat0-np.asarray(tlat))/dj).astype(int);ix=np.rint((np.asarray(tlon)-lon0)/di).astype(int);yy,xx=np.meshgrid(iy,ix,indexing="ij");indices=(yy*ni+xx).ravel().tolist();x=np.asarray(eccodes.codes_get_double_elements(message,"values",indices),np.float32).reshape(len(tlat),len(tlon))
 finally:eccodes.codes_release(message)
 if product in ("merged_reflectivity","lowest_reflectivity"):
  x[x<=-900]=np.nan;x[(x<=-90)&np.isfinite(x)]=0
 else:
  x[x<=-3]=np.nan;x[(x<0)&np.isfinite(x)]=0
 return x
def decode_task(task):
 path,product,tlat,tlon=task
 return decode_target(Path(path),product,tlat,tlon)
def stats(x,prefix,threshold=None):
 q=np.isfinite(x); vals=x[q]; out={f"{prefix}_p90":float(np.nanpercentile(vals,90)) if len(vals) else np.nan,f"{prefix}_max":float(np.nanmax(vals)) if len(vals) else np.nan}
 if threshold is not None:out[f"{prefix}_cov20" if "reflectivity" not in prefix and prefix in ("merged","lowest") else f"{prefix}_cov"]=float(np.nanmean(x>=threshold))
 return out

def materialize():
 r=rows(); sm=pd.read_csv(ROOT/"source_manifest.csv"); mp={(x.row_id,x.product,x.source_role):x for x in sm.itertuples(index=False)}; cache=pd.read_csv("artifacts/stage_8/development_features/development_feature_manifest.csv").set_index("row_id"); events={}; records=[]; precursor=[]
 executor=concurrent.futures.ProcessPoolExecutor(max_workers=8)
 for num,rr in enumerate(r.itertuples(index=False),1):
  if rr.event_id not in events:
   z=np.load(event_path(rr.event_id,rr.radar_source_path)); events[rr.event_id]=(_downsample_max(z["rate_mm_hr"],2),pd.to_datetime(z["times"],utc=True),z["latitude"],z["longitude"])
  rate,times,lat,lon=events[rr.event_id];lat=lat[:rate.shape[1]*2:2] if len(lat)>rate.shape[1] else lat;lon=lon[:rate.shape[2]*2:2] if len(lon)>rate.shape[2] else lon;y0,y1,x0,x1=map(int,[rr.tile_y_min,rr.tile_y_max,rr.tile_x_min,rr.tile_x_max]);tlat=lat[y0:y1+1];tlon=lon[x0:x1+1];bbox=[tlon.min()-.03,tlat.min()-.03,tlon.max()+.03,tlat.max()+.03]
  tasks=[(product,role,mp[(rr.row_id,product,role)].local_path) for product in PRODUCTS for role in ("current","baseline_30min")]
  values=list(executor.map(decode_task,[(path,product,tlat,tlon) for product,role,path in tasks]));fields={(product,role):field for (product,role,_),field in zip(tasks,values)};valid=[np.isfinite(v).mean() for v in values]
  issue=pd.Timestamp(rr.issue_time_utc);idx=times.get_indexer([issue])[0];cur_rate=rate[idx,y0:y1+1,x0:x1+1]; z=np.load(cache.loc[rr.row_id,"cache_path"]);py=z["pysteps_rate"].astype(float);target=z["target_rate"].astype(float)
  m=fields[("merged_reflectivity","current")];mb=fields[("merged_reflectivity","baseline_30min")];lo=fields[("lowest_reflectivity","current")];lob=fields[("lowest_reflectivity","baseline_30min")];e30=fields[("echo_top_30","current")];e30b=fields[("echo_top_30","baseline_30min")];e50=fields[("echo_top_50","current")];e50b=fields[("echo_top_50","baseline_30min")];vil=fields[("vil","current")];vilb=fields[("vil","baseline_30min")]
  mg=m-mb;lg=lo-lob;g30=e30-e30b;g50=e50-e50b;vg=vil-vilb;dry=cur_rate<=.1; elevated=(m>=REFLECTIVITY_DBZ)&dry&((e30>0)|(e50>0)|(vil>0));vertical_growth=(g30>=ECHOTOP_GROWTH_KM)|(g50>=ECHOTOP_GROWTH_KM)|(vg>=VIL_GROWTH);refl_only=(m>=REFLECTIVITY_DBZ)&dry
  base={"row_id":rr.row_id,"role":rr.role,"event_id":rr.event_id,"independent_system_group":rr.independent_system_group,"issue_time_utc":issue.isoformat(),"mrms_valid_fraction":min(valid),"current_precip_wet_fraction":float(np.nanmean(cur_rate>.1)),
   "merged_cov20":float(np.nanmean(m>=REFLECTIVITY_DBZ)),"merged_p90":float(np.nanpercentile(m,90)),"merged_max":float(np.nanmax(m)),"merged_growth_cov":float(np.nanmean(mg>=REFLECTIVITY_GROWTH_DBZ)),"merged_growth_p90":float(np.nanpercentile(mg,90)),
   "lowest_cov20":float(np.nanmean(lo>=REFLECTIVITY_DBZ)),"lowest_p90":float(np.nanpercentile(lo,90)),"lowest_max":float(np.nanmax(lo)),"lowest_growth_cov":float(np.nanmean(lg>=REFLECTIVITY_GROWTH_DBZ)),"lowest_growth_p90":float(np.nanpercentile(lg,90)),
   "et30_cov":float(np.nanmean(e30>0)),"et30_p90":float(np.nanpercentile(e30,90)),"et30_max":float(np.nanmax(e30)),"et30_growth_cov":float(np.nanmean(g30>=ECHOTOP_GROWTH_KM)),"et30_growth_p90":float(np.nanpercentile(g30,90)),
   "et50_cov":float(np.nanmean(e50>0)),"et50_p90":float(np.nanpercentile(e50,90)),"et50_max":float(np.nanmax(e50)),"et50_growth_cov":float(np.nanmean(g50>=ECHOTOP_GROWTH_KM)),"et50_growth_p90":float(np.nanpercentile(g50,90)),
   "vil_cov":float(np.nanmean(vil>0)),"vil_p90":float(np.nanpercentile(vil,90)),"vil_growth_cov":float(np.nanmean(vg>=VIL_GROWTH)),"vil_growth_p90":float(np.nanpercentile(vg,90)),"reflectivity_only_fraction":float(np.nanmean(refl_only)),"elevated_echo_without_surface_rain_fraction":float(np.nanmean(elevated))}
  eventual=np.any(target>.1,axis=0)&dry; pyany=np.any(py>.1,axis=0);miss=eventual&~pyany;success=eventual&pyany
  def frac(mask,den):return float(np.sum(mask&den)/np.sum(den)) if np.any(den) else np.nan
  onset=np.argmax(target>.1,axis=0)*6+6; leadmask=eventual&elevated
  precursor.append({"row_id":rr.row_id,"role":rr.role,"independent_system_group":rr.independent_system_group,"future_initiation_pixels":int(eventual.sum()),"missed_pixels":int(miss.sum()),"miss_merged_precursor":frac(m>=REFLECTIVITY_DBZ,miss),"miss_et30_precursor":frac(e30>0,miss),"miss_et50_precursor":frac(e50>0,miss),"miss_vil_precursor":frac(vil>0,miss),"miss_vertical_growth":frac(vertical_growth,miss),"success_elevated_precursor":frac(elevated,success),"miss_category_A_none":frac(~refl_only&~vertical_growth,miss),"miss_category_B_reflectivity_only":frac(refl_only&~vertical_growth,miss),"miss_category_C_vertical_growth":frac(vertical_growth,miss),"miss_category_D_echo_outside_precip":frac(refl_only,miss),**{f"precursor_onset_{b}":frac(leadmask&(onset>lo0)&(onset<=hi),leadmask) for b,lo0,hi in [("le15",0,15),("15_30",15,30),("30_60",30,60),("gt60",60,120)]}})
  for j in range(20):records.append({**base,"lead_minutes":6*(j+1),"lead_fraction":(j+1)/20,"pysteps_wet_fraction":float(np.nanmean(py[j]>.1))})
  if num%20==0:print(f"[stage12-materialize] {num}/{len(r)}",flush=True)
 executor.shutdown();pd.DataFrame(records).to_csv(ROOT/"radar_descriptors_by_row_lead.csv",index=False);pd.DataFrame(precursor).to_csv(ROOT/"precursor_pixel_diagnostics.csv",index=False)
 pd.DataFrame([{"availability_gate":AVAILABILITY,"reflectivity_threshold_dbz":REFLECTIVITY_DBZ,"reflectivity_growth_dbz_30min":REFLECTIVITY_GROWTH_DBZ,"echo_top_growth_km_30min":ECHOTOP_GROWTH_KM,"vil_growth_kg_m2_30min":VIL_GROWTH,"elevated_echo_without_surface_rain":"merged reflectivity >=20 dBZ AND PrecipRate <=0.1 AND (EchoTop30>0 OR EchoTop50>0 OR VIL>0)","frozen_before_truth_join":True}]).to_csv(ROOT/"feature_predeclaration.csv",index=False)

def auc(y,p):
 n1=y.sum();n0=len(y)-n1;return np.nan if not n1 or not n0 else float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def fit(x,y,g):
 mean=x.mean(0);std=x.std(0);std[std==0]=1;z=(x-mean)/std;X=np.c_[np.ones(len(x)),z];counts=pd.Series(g).value_counts();w=np.array([1/counts[q] for q in g]);w/=w.sum()
 def fg(b):p=1/(1+np.exp(-np.clip(X@b,-30,30)));loss=-np.sum(w*(y*np.log(np.clip(p,1e-8,1))+(1-y)*np.log(np.clip(1-p,1e-8,1))))+np.sum(b[1:]**2);gr=X.T@(w*(p-y));gr[1:]+=2*b[1:];return loss,gr
 res=minimize(lambda b:fg(b),np.zeros(X.shape[1]),jac=True,method="L-BFGS-B");return mean,std,res.x,res.success
def predict(x,m,s,b):return 1/(1+np.exp(-np.clip(np.c_[np.ones(len(x)),(x-m)/s]@b,-30,30)))
def metric(y,p):
 q=p>=.5;tp=np.sum(q&(y==1));fp=np.sum(q&(y==0));fn=np.sum(~q&(y==1));pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0;return {"auc":auc(y,p),"precision":pr,"recall":re,"f1":2*pr*re/(pr+re) if pr+re else 0,"brier":np.mean((p-y)**2)}
def evaluate():
 d=pd.read_csv(ROOT/"radar_descriptors_by_row_lead.csv");o=pd.read_csv("artifacts/stage_9/source_advantage/oracle_source_advantage_by_row_lead.csv");o=o[o.collection.eq("stage8_nested_development")][["row_id","lead_minutes","pysteps_failure_category"]];d=d.merge(o,on=["row_id","lead_minutes"],validate="one_to_one");d["failure_target"]=d.pysteps_failure_category.isin(["no_initiation_signal","growth_underprediction"]).astype(int);p=d[d.role.eq("positive_initiation")&d.mrms_valid_fraction.ge(.9)].reset_index(drop=True);g=p.independent_system_group.to_numpy();y=p.failure_target.to_numpy();rng=np.random.default_rng(1201);models={"context_only":CONTEXT,"context_plus_richer_radar":CONTEXT+RADAR_FEATURES};out=[];fits=[]
 for name,features in models.items():
  x=p[features].to_numpy();pred=np.zeros(len(p));removed=np.zeros(len(p));sg=np.zeros(len(p));se=np.zeros(len(p));gp=rng.permutation(len(p))
  for hold in sorted(set(g)):
   tr=g!=hold;te=~tr;m,s,b,ok=fit(x[tr],y[tr],g[tr]);pred[te]=predict(x[te],m,s,b);a=x[te].copy();bb=x[te].copy();c=x[te].copy()
   if name!="context_only":
    ix=list(range(len(CONTEXT),len(features)));a[:,ix]=m[ix];bb[:,ix]=bb[rng.permutation(te.sum())][:,ix];c[:,ix]=x[gp[:te.sum()]][:,ix]
   removed[te]=predict(a,m,s,b);se[te]=predict(bb,m,s,b);sg[te]=predict(c,m,s,b);fits.append({"model":name,"held_out_system":hold,"success":bool(ok),"training_systems":"|".join(sorted(set(g[tr]))),"normalization_training_only":True})
  for mode,v in [("normal",pred),("radar_removed",removed),("radar_shuffled_global",sg),("radar_shuffled_within_event",se)]:
   for i,z in enumerate(v):out.append({"model":name,"destruction":mode,"row_index":i,"probability":z})
 pr=pd.DataFrame(out).merge(p.reset_index().rename(columns={"index":"row_index"})[["row_index","row_id","independent_system_group","lead_minutes","failure_target","pysteps_failure_category"]],on="row_index");pr.to_csv(ROOT/"loeo_predictions.csv",index=False);pd.DataFrame(fits).to_csv(ROOT/"loeo_fits.csv",index=False);ev=[]
 for (m,z,e),t in pr.groupby(["model","destruction","independent_system_group"]):ev.append({"model":m,"destruction":z,"event":e,"samples":len(t),**metric(t.failure_target.to_numpy(),t.probability.to_numpy())})
 ev=pd.DataFrame(ev);ev.to_csv(ROOT/"loeo_metrics_by_event.csv",index=False);ev.groupby(["model","destruction"],as_index=False).mean(numeric_only=True).to_csv(ROOT/"loeo_metrics_event_macro.csv",index=False)
 p.groupby("pysteps_failure_category",as_index=False)[RADAR_FEATURES].mean().to_csv(ROOT/"descriptors_by_failure_category.csv",index=False);d[d.role.eq("hard_negative")].groupby("independent_system_group",as_index=False)[RADAR_FEATURES].mean().to_csv(ROOT/"hard_negative_descriptors_by_event.csv",index=False)
 px=pd.read_csv(ROOT/"precursor_pixel_diagnostics.csv");pxp=px[px.role.eq("positive_initiation")];precursor_cols=[c for c in px if c.startswith(("miss_","success_","precursor_"))];pxp.groupby("independent_system_group",as_index=False)[precursor_cols].mean().to_csv(ROOT/"precursor_diagnostics_by_event.csv",index=False)
 trend=["merged_growth_cov","merged_growth_p90","lowest_growth_cov","lowest_growth_p90","et30_growth_cov","et30_growth_p90","et50_growth_cov","et50_growth_p90","vil_growth_cov","vil_growth_p90"]
 gm=p[p.pysteps_failure_category.isin(["growth_underprediction","other_or_adequate"])].groupby(["independent_system_group","pysteps_failure_category"])[trend].mean().unstack();growth=[]
 for event,row in gm.iterrows():growth.append({"event":event,**{f"delta_{f}_growth_minus_adequate":row[(f,"growth_underprediction")]-row[(f,"other_or_adequate")] for f in trend}})
 pd.DataFrame(growth).to_csv(ROOT/"growth_vs_adequate_by_event.csv",index=False)
 assoc=[]
 for f in RADAR_FEATURES:
  vals=[]
  for e,t in p.groupby("independent_system_group"):v=spearmanr(t[f],t.failure_target).statistic;vals.append(v);assoc.append({"feature":f,"event":e,"spearman":v})
  assoc.append({"feature":f,"event":"CONSISTENCY","positive_systems":int(np.sum(np.array(vals)>0)),"negative_systems":int(np.sum(np.array(vals)<0))})
 pd.DataFrame(assoc).to_csv(ROOT/"descriptor_event_consistency.csv",index=False)
 # Frozen group-removal probes use the same full-model fits, with each group replaced by training means.
 ab=[];features=CONTEXT+RADAR_FEATURES;x=p[features].to_numpy()
 for hold in sorted(set(g)):
  tr=g!=hold;te=~tr;m,s,b,_=fit(x[tr],y[tr],g[tr])
  for group,names in GROUPS.items():xx=x[te].copy();ix=[features.index(q) for q in names];xx[:,ix]=m[ix];ab.append({"held_out_system":hold,"removed_group":group,**metric(y[te],predict(xx,m,s,b))})
 pd.DataFrame(ab).to_csv(ROOT/"product_group_removal_by_event.csv",index=False)
 full=pr[(pr.model=="context_plus_richer_radar")&(pr.destruction=="normal")];ctx=pr[(pr.model=="context_only")&(pr.destruction=="normal")];cats=[]
 calibration=[]
 for model,t in pd.concat([full,ctx]).groupby("model"):
  tt=t.copy();tt["bin"]=pd.cut(tt.probability,np.linspace(0,1,11),include_lowest=True)
  for b,q in tt.groupby("bin",observed=True):calibration.append({"model":model,"bin":str(b),"samples":len(q),"mean_probability":q.probability.mean(),"observed_frequency":q.failure_target.mean()})
 pd.DataFrame(calibration).to_csv(ROOT/"loeo_calibration.csv",index=False)
 for cat in sorted(p.pysteps_failure_category.unique()):
  aa=[];cc=[]
  for e,t in full.groupby("independent_system_group"):
   yy=t.pysteps_failure_category.eq(cat).astype(int).to_numpy();aa.append(auc(yy,t.probability.to_numpy()));ct=ctx[ctx.independent_system_group.eq(e)];cc.append(auc(yy,ct.probability.to_numpy()))
  cats.append({"category":cat,"richer_radar_event_macro_auc":np.nanmean(aa),"context_event_macro_auc":np.nanmean(cc)})
 pd.DataFrame(cats).to_csv(ROOT/"failure_category_discrimination.csv",index=False);macro=pd.read_csv(ROOT/"loeo_metrics_event_macro.csv");n=macro[(macro.model=="context_plus_richer_radar")&(macro.destruction=="normal")].iloc[0];c=macro[(macro.model=="context_only")&(macro.destruction=="normal")].iloc[0];pv=ev[ev.destruction.eq("normal")].pivot(index="event",columns="model",values="auc");gain=pv.context_plus_richer_radar-pv.context_only;destroy=macro[(macro.model=="context_plus_richer_radar")&macro.destruction.ne("normal")].set_index("destruction");prom=bool(n.auc>c.auc+.02 and (gain>0).sum()>=4 and n.auc>destroy.auc.max()+.01);classification="PROMISING" if prom else ("WEAK / REGIME-DEPENDENT" if n.auc>c.auc and (gain>0).sum()>=2 else "NOT USEFUL")
 decision={"classification":classification,"context_auc":c.auc,"richer_radar_auc":n.auc,"auc_increment":n.auc-c.auc,"systems_improved":int((gain>0).sum()),"destruction_max_auc":destroy.auc.max(),"correction_experiment_justified":prom,"promotable_model_trained":False,"sealed_final_accessed":False};(ROOT/"stage12_decision.json").write_text(json.dumps(decision,indent=2));outputs={x.name:sha256_file(x) for x in ROOT.glob("*.csv")};(ROOT/"stage12_manifest.json").write_text(json.dumps({"status":"DIAGNOSTIC COMPLETE","rows":d.row_id.nunique(),"usable_positive_rows":p.row_id.nunique(),"systems":p.independent_system_group.nunique(),"promotable_model_trained":False,"sealed_final_accessed":False,"outputs":outputs,"decision_sha256":sha256_file(ROOT/"stage12_decision.json")},indent=2));print(json.dumps(decision,indent=2))

def main():
 q=argparse.ArgumentParser();q.add_argument("phase",choices=["audit","download","materialize","evaluate","all"],nargs="?",default="all");a=q.parse_args()
 if a.phase in ("audit","all"):audit()
 if a.phase in ("download","all"):download()
 if a.phase in ("materialize","all"):materialize()
 if a.phase in ("evaluate","all"):evaluate()
if __name__=="__main__":main()
