"""Audit historical HRRR wrfsubhf coverage for consumed Stage 8 cycles."""
from __future__ import annotations
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
import pandas as pd
ROOT=Path("artifacts/stage_10b"); TOKENS=("APCP:","PRATE:","REFC:","REFD:1000 m above ground")
def fetch(task):
 t,fh=task; base=f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{t:%Y%m%d}/conus/hrrr.t{t:%H}z.wrfsubhf{fh:02d}.grib2"; out={"cycle":t.isoformat(),"forecast_hour_file":fh,"object_url":base,"idx_url":base+".idx","object_exists":False,"idx_exists":False,"object_bytes":None,"idx_sha256":None,"messages":[]}
 try:
  req=urllib.request.Request(base,method="HEAD");res=urllib.request.urlopen(req,timeout=45);out["object_exists"]=res.status==200;out["object_bytes"]=int(res.headers.get("Content-Length",0))
  raw=urllib.request.urlopen(base+".idx",timeout=45).read();out["idx_exists"]=True;out["idx_sha256"]=hashlib.sha256(raw).hexdigest();lines=raw.decode().splitlines();out["messages"]=[x for x in lines if any(tok in x for tok in TOKENS)]
 except Exception as e:out["error"]=repr(e)
 return out
def run():
 ROOT.mkdir(exist_ok=True);h=pd.read_csv("artifacts/stage_8/qualification/stage8_hrrr_causal_records.csv");cycles=sorted(pd.to_datetime(h.hrrr_cycle_issue_time_utc,utc=True).unique());tasks=[(pd.Timestamp(t),fh) for t in cycles for fh in (1,2)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex: results=list(ex.map(fetch,tasks))
 rows=[]
 for r in results:
  base={k:v for k,v in r.items() if k!="messages"}
  if r["messages"]:
   for line in r["messages"]:rows.append({**base,"message":line})
  else:rows.append({**base,"message":""})
 pd.DataFrame(rows).to_csv(ROOT/"wrfsubhf_archive_audit.csv",index=False)
 files=pd.DataFrame([{k:v for k,v in r.items() if k!="messages"}|{"candidate_message_count":len(r["messages"])} for r in results]);files.to_csv(ROOT/"wrfsubhf_file_coverage.csv",index=False)
 summary={"required_cycles":len(cycles),"required_files":len(tasks),"available_files":int((files.object_exists&files.idx_exists).sum()),"complete_cycles":int(files.assign(ok=files.object_exists&files.idx_exists).groupby("cycle").ok.all().sum()),"years":sorted({pd.Timestamp(x).year for x in cycles}),"sealed_final_accessed":False};(ROOT/"wrfsubhf_audit_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":run()
