"""Download selected quarter-hour HRRR fields after the Stage 10B archive gate."""
from __future__ import annotations
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
import eccodes,numpy as np,pandas as pd
from ..data.manifest import sha256_file
ROOT=Path("artifacts/stage_10b");CACHE=ROOT/"selected_messages";VARS={"APCP":"APCP:surface:","PRATE":"PRATE:surface:","REFC":"REFC:entire atmosphere:"}
INDEXES={}
def get(url):return urllib.request.urlopen(url,timeout=90).read()
def one(task):
 t,fh,var,minute=task;base,lines=INDEXES[(t.isoformat(),fh)]; token=VARS[var]
 candidates=[]
 for i,line in enumerate(lines):
  if token in line and f":{minute} min fcst:" in line or (token in line and var=="APCP" and line.endswith(f"-{minute} min acc fcst:")):
   candidates.append((i,line))
 if len(candidates)!=1:raise RuntimeError((t,fh,var,minute,candidates))
 i,line=candidates[0];start=int(line.split(":")[1]);end=int(lines[i+1].split(":")[1])-1 if i+1<len(lines) else None;path=CACHE/f"{t:%Y%m%dT%H}_{fh:02d}_{minute:03d}_{var}.grib2"
 if not path.exists():
  req=urllib.request.Request(base,headers={"Range":f"bytes={start}-{end}" if end else f"bytes={start}-"})
  for attempt in range(5):
   try:path.write_bytes(urllib.request.urlopen(req,timeout=180).read());break
   except Exception:
    if attempt==4:raise
    time.sleep(2**attempt)
 return {"cycle":t.isoformat(),"forecast_hour_file":fh,"offset_minutes":minute,"variable":var,"record":line,"path":path.as_posix(),"sha256":sha256_file(path),"bytes":path.stat().st_size,"source_url":base,"byte_start":start,"byte_end":end}
def metadata(path):
 with Path(path).open("rb") as f:m=eccodes.codes_grib_new_from_file(f)
 try:
  v=np.asarray(eccodes.codes_get_values(m));return {k:eccodes.codes_get(m,k) for k in ["shortName","name","units","stepType","stepRange","startStep","endStep","validityDate","validityTime"]}|{"finite_fraction":float(np.isfinite(v).mean()),"minimum":float(np.nanmin(v)),"maximum":float(np.nanmax(v)),"mean":float(np.nanmean(v))}
 finally:eccodes.codes_release(m)
def run():
 CACHE.mkdir(parents=True,exist_ok=True);h=pd.read_csv("artifacts/stage_8/qualification/stage8_hrrr_causal_records.csv");cycles=sorted(pd.to_datetime(h.hrrr_cycle_issue_time_utc,utc=True).unique());tasks=[]
 for tt in cycles:
  t=pd.Timestamp(tt)
  for fh,mins in [(1,(15,30,45,60)),(2,(75,90,105,120))]:
   for minute in mins:
    for var in VARS:tasks.append((t,fh,var,minute))
 index_tasks=sorted({(t,fh) for t,fh,_,_ in tasks},key=lambda x:(x[0],x[1]))
 def load_index(pair):
  t,fh=pair;base=f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{t:%Y%m%d}/conus/hrrr.t{t:%H}z.wrfsubhf{fh:02d}.grib2";return (t.isoformat(),fh),(base,get(base+".idx").decode().splitlines())
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex: INDEXES.update(dict(ex.map(load_index,index_tasks)))
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex: rows=list(ex.map(one,tasks))
 manifest=pd.DataFrame(rows);manifest.to_csv(ROOT/"wrfsubhf_selected_message_manifest.csv",index=False)
 # Smoke gate spans three systems/years and all variables/quarter-hour changes.
 samples=[]
 for cycle in [cycles[0],cycles[len(cycles)//2],cycles[-1]]:
  sub=manifest[manifest.cycle.eq(pd.Timestamp(cycle).isoformat()) & manifest.offset_minutes.isin([15,30,45,60])]
  for r in sub.itertuples(index=False):samples.append({"cycle":r.cycle,"variable":r.variable,"offset_minutes":r.offset_minutes,**metadata(r.path)})
 smoke=pd.DataFrame(samples);smoke.to_csv(ROOT/"representative_decode_smoke.csv",index=False);changing=smoke.groupby(["cycle","variable"])["mean"].nunique().gt(1).all();gate={"files":len(manifest),"expected":len(tasks),"all_hashes_present":bool(manifest.sha256.str.len().eq(64).all()),"all_smoke_finite":bool(smoke.finite_fraction.eq(1).all()),"quarter_hour_values_change":bool(changing),"passed":bool(len(manifest)==len(tasks) and changing and smoke.finite_fraction.eq(1).all()),"sealed_final_accessed":False};(ROOT/"representative_smoke_gate.json").write_text(json.dumps(gate,indent=2));print(json.dumps(gate,indent=2))
if __name__=="__main__":run()
