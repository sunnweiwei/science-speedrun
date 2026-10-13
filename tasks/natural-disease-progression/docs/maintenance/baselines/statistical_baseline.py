"""Empirical-Bayes diagnosis-burden baseline for longitudinal-v2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

HORIZONS=tuple(range(30,361,30))+(365,)


def _feature_rows(input_dir: Path, edges: list[float]) -> tuple[list[dict],dict[str,tuple[int,Counter]]]:
    samples=sorted(pq.read_table(input_dir/"samples.parquet").to_pylist(),key=lambda x:x["sample_id"])
    db=duckdb.connect(); rows=db.execute("""SELECT sample_id,count(*) n,
      list(left(code,1)) FILTER (WHERE NOT starts_with(code,'ICD9:')) chapters
      FROM read_parquet(?) WHERE event_type='diagnosis' GROUP BY sample_id""",
      [str(input_dir/"events.parquet")]).fetchall(); db.close()
    return samples,{sid:(int(n),Counter(ch or [])) for sid,n,ch in rows}


def fit(materialized: Path, output: Path, alpha: float=150.0) -> dict:
    manifest=json.loads((materialized/"manifest.json").read_text()); diseases=manifest["scored_diseases"]
    db=duckdb.connect(); events=str(materialized/"train/events.parquet"); samples=str(materialized/"train/samples.parquet"); labels=str(materialized/"train/label_windows.parquet")
    edges=[float(x) for x in db.execute("""WITH x AS (SELECT sample_id,count(*) n FROM read_parquet(?)
      WHERE event_type='diagnosis' GROUP BY sample_id) SELECT quantile_cont(n,[.2,.4,.6,.8]) FROM x""",[events]).fetchone()[0]]
    db.execute(f"""CREATE TEMP TABLE feat AS WITH diagnosis AS (
      SELECT sample_id,count(*) n FROM read_parquet('{events}') WHERE event_type='diagnosis' GROUP BY 1)
      SELECT s.sample_id,s.gender,CASE WHEN approx_age<45 THEN 0 WHEN approx_age<65 THEN 1 WHEN approx_age<80 THEN 2 ELSE 3 END age_bin,
        CASE WHEN coalesce(d.n,0)<={edges[0]} THEN 0 WHEN d.n<={edges[1]} THEN 1 WHEN d.n<={edges[2]} THEN 2 WHEN d.n<={edges[3]} THEN 3 ELSE 4 END burden
      FROM read_parquet('{samples}') s LEFT JOIN diagnosis d USING(sample_id)""")
    db.execute(f"""CREATE TEMP TABLE chapter AS SELECT sample_id,left(code,1) chapter,count(*) n
      FROM read_parquet('{events}') WHERE event_type='diagnosis' AND NOT starts_with(code,'ICD9:') GROUP BY 1,2""")
    D,H=len(diseases),len(HORIZONS); di={d:i for i,d in enumerate(diseases)}
    # Parent rates: age/sex/burden; local rates add candidate-disease chapter count.
    parents:dict[str,np.ndarray]={}; locals_:dict[str,np.ndarray]={}
    for k,h in enumerate(HORIZONS):
        parent=db.execute(f"""SELECT f.gender,f.age_bin,f.burden,l.disease_code,
          count(*) FILTER(WHERE l.disease_valid_{h}) n,count(*) FILTER(WHERE l.disease_label_{h}) p
          FROM read_parquet('{labels}') l JOIN feat f USING(sample_id)
          WHERE l.diagnosis_observation_end>=l.cutoff+INTERVAL 365 DAY
            AND l.death_observation_end>=l.cutoff+INTERVAL 365 DAY GROUP BY 1,2,3,4""").fetchall()
        for sex,age,burden,d,n,p in parent:
            key=f"{sex}:{age}:{burden}"; parents.setdefault(key,np.full((D,H),np.nan,np.float32))
            parents[key][di[d],k]=(p+1)/(n+2)
        local=db.execute(f"""SELECT f.gender,f.age_bin,f.burden,
          least(2,coalesce(c.n,0)) chapter_count,l.disease_code,
          count(*) FILTER(WHERE l.disease_valid_{h}) n,count(*) FILTER(WHERE l.disease_label_{h}) p
          FROM read_parquet('{labels}') l JOIN feat f USING(sample_id)
          LEFT JOIN chapter c ON l.sample_id=c.sample_id AND left(l.disease_code,1)=c.chapter
          WHERE l.diagnosis_observation_end>=l.cutoff+INTERVAL 365 DAY
            AND l.death_observation_end>=l.cutoff+INTERVAL 365 DAY GROUP BY 1,2,3,4,5""").fetchall()
        for sex,age,burden,cc,d,n,p in local:
            parent_key=f"{sex}:{age}:{burden}"; key=f"{parent_key}:{cc}"
            locals_.setdefault(key,np.full((D,H),np.nan,np.float32))
            prior=parents[parent_key][di[d],k]; locals_[key][di[d],k]=(p+alpha*prior)/(n+alpha)
    # Global fallback and monotonic cumulative risks.
    global_rate=np.empty((D,H),np.float32)
    for k,h in enumerate(HORIZONS):
        rows=db.execute(f"""SELECT disease_code,count(*) FILTER(WHERE disease_valid_{h}),
          count(*) FILTER(WHERE disease_label_{h}) FROM read_parquet('{labels}')
          WHERE diagnosis_observation_end>=cutoff+INTERVAL 365 DAY
            AND death_observation_end>=cutoff+INTERVAL 365 DAY GROUP BY 1""").fetchall()
        for d,n,p in rows: global_rate[di[d],k]=(p+1)/(n+2)
    global_rate=np.maximum.accumulate(global_rate,axis=1)
    for mapping in (parents,locals_):
        for key,a in mapping.items():
            a[np.isnan(a)]=global_rate[np.isnan(a)]; mapping[key]=np.maximum.accumulate(a,axis=1)
    # Death retains the already strong demographic checkpoint behavior.
    first=diseases[0]; death={}
    for sex in ("F","M"):
      for age in range(4):
       vals=[]
       for h in HORIZONS:
        n,p=db.execute(f"""SELECT count(*) FILTER(WHERE death_valid_{h}),count(*) FILTER(WHERE death_label_{h})
          FROM read_parquet('{labels}') l JOIN feat f USING(sample_id)
          WHERE disease_code=? AND gender=? AND age_bin=?
            AND diagnosis_observation_end>=cutoff+INTERVAL 365 DAY AND death_observation_end>=cutoff+INTERVAL 365 DAY""",[first,sex,age]).fetchone(); vals.append((p+1)/(n+2))
       death[f"{sex}:{age}"]=np.maximum.accumulate(vals).astype(np.float32)
    db.close(); output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"schema":"ndp-empirical-bayes/v2","diseases":diseases,"horizons":HORIZONS,"edges":edges,"alpha":alpha,"global":global_rate,"parents":parents,"locals":locals_,"death":death},output)
    report={"alpha":alpha,"burden_edges":edges,"parent_groups":len(parents),"local_groups":len(locals_),"fit_source":"complete-year train labels only"}
    output.with_suffix(".training.json").write_text(json.dumps(report,indent=2)+"\n"); return report


def predict(input_dir:Path,checkpoint:Path,output:Path)->dict:
    s=torch.load(checkpoint,map_location="cpu",weights_only=False); samples,feat=_feature_rows(input_dir,s["edges"]); out=[]
    D,H=len(s["diseases"]),len(HORIZONS)
    for row in samples:
        age=row["approx_age"]; ab=0 if age<45 else 1 if age<65 else 2 if age<80 else 3
        n,ch=feat.get(row["sample_id"],(0,Counter())); burden=int(np.searchsorted(s["edges"],n,side="left")); parent=f'{row["gender"]}:{ab}:{burden}'
        p=np.empty((D,H),np.float32)
        for j,d in enumerate(s["diseases"]):
            key=f"{parent}:{min(2,ch[d[0]])}"; p[j]=s["locals"].get(key,s["parents"].get(parent,s["global"]))[j]
        death=s["death"].get(f'{row["gender"]}:{ab}',np.zeros(H,np.float32))
        out.append({"sample_id":row["sample_id"],"disease_risk":p.reshape(-1).tolist(),"death_risk":death.tolist()})
    schema=pa.schema([pa.field("sample_id",pa.string()),pa.field("disease_risk",pa.list_(pa.float32(),D*H)),pa.field("death_risk",pa.list_(pa.float32(),H))])
    output.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(out,schema=schema),output,compression="zstd");return {"samples":len(out)}


if __name__=="__main__":
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("fit");p.add_argument("--materialized",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--alpha",type=float,default=150)
    p=sp.add_parser("predict");p.add_argument("--input",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();print(json.dumps(fit(a.materialized,a.output,a.alpha) if a.cmd=="fit" else predict(a.input,a.checkpoint,a.output),indent=2))
