"""Reproduce v1/v2 learning-signal aggregates without exporting patient rows."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import duckdb

def summarize(db,root:Path,split:str,protocol:str)->dict:
 base=root/"train" if split=="train" else root/"public"/split
 samples=str(base/"samples.parquet");events=str(base/"events.parquet")
 stay="stay_key" if protocol=="v2" else "CAST(hadm_id AS VARCHAR)"
 row=db.execute(f"""WITH h AS (SELECT sample_id,count(*) n,count(DISTINCT {stay}) FILTER(WHERE {stay} IS NOT NULL) stays,
   count(*) FILTER(WHERE event_type='lab') labs,count(*) FILTER(WHERE event_type='lab' AND try_cast(value AS DOUBLE) IS NOT NULL) numeric_labs,
   count(*) FILTER(WHERE event_type='diagnosis') diagnoses,count(*) FILTER(WHERE event_type IN ('prescription','administration')) medications
   FROM read_parquet(?) GROUP BY 1), all_samples AS (SELECT s.sample_id,coalesce(h.n,0) n,coalesce(h.stays,0) stays,coalesce(h.labs,0) labs,coalesce(h.numeric_labs,0) numeric_labs,coalesce(h.diagnoses,0) diagnoses,coalesce(h.medications,0) medications FROM read_parquet(?) s LEFT JOIN h USING(sample_id))
   SELECT count(*),quantile_cont(n,[0,.1,.25,.5,.75,.9,.95,.99,1]),count(*) FILTER(WHERE n>512),count(*) FILTER(WHERE stays>=2),count(*) FILTER(WHERE labs>0),sum(numeric_labs),count(*) FILTER(WHERE medications>0),count(*) FILTER(WHERE diagnoses>0) FROM all_samples""",[events,samples]).fetchone()
 return {"samples":row[0],"event_quantiles":{"p0":row[1][0],"p10":row[1][1],"p25":row[1][2],"p50":row[1][3],"p75":row[1][4],"p90":row[1][5],"p95":row[1][6],"p99":row[1][7],"p100":row[1][8]},"samples_over_512_events":row[2],"samples_with_two_or_more_event_stays":row[3],"samples_with_labs":row[4],"numeric_lab_rows":row[5],"samples_with_medications":row[6],"samples_with_diagnoses":row[7]}

def run(v1:Path,v2:Path,include_test_truth:bool)->dict:
 db=duckdb.connect();out={"schema":"ndp-learning-signal-audit/v2","v1":{},"longitudinal_v2":{}}
 for name,root in (("v1",v1),("longitudinal_v2",v2)):
  protocol="v2" if name.endswith("v2") else "v1"
  for split in ("train","validation","test"):out[name][split]=summarize(db,root,split,protocol)
  # Aggregate target support; test truth is opt-in because it compromises a pristine holdout.
  for split in ("train","validation")+(("test",) if include_test_truth else ()):
   path=root/("train" if split=="train" else f"trusted/{split}")/"label_windows.parquet"
   h=365
   row=db.execute(f"""WITH x AS (SELECT sample_id,max(CAST(disease_label_{h} AS INTEGER)) any_event,max(CAST(disease_valid_{h} AS INTEGER)) any_valid FROM read_parquet(?) GROUP BY 1) SELECT count(*),sum(any_event),sum(any_valid) FROM x""",[str(path)]).fetchone()
   out[name][split]["target_365"]={"samples_in_label_grid":row[0],"samples_with_any_scored_event":row[1],"samples_with_any_valid_target":row[2]}
 db.close();return out

if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--v1",type=Path,required=True);ap.add_argument("--v2",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--include-test-truth",action="store_true");a=ap.parse_args();r=run(a.v1,a.v2,a.include_test_truth);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
