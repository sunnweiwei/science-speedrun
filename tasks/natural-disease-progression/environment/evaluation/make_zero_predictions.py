"""Create the deterministic all-zero sanity baseline."""
import argparse,json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
m=json.loads((a.input/"manifest.json").read_text());d=len(m["scored_diseases"]);h=len(m["horizons_days"])
ids=pq.read_table(a.input/"samples.parquet",columns=["sample_id"])["sample_id"].to_pylist()
schema=pa.schema([pa.field("sample_id",pa.string()),pa.field("disease_risk",pa.list_(pa.float32(),d*h)),pa.field("death_risk",pa.list_(pa.float32(),h))])
row_d=[0.0]*(d*h);row_h=[0.0]*h;a.output.parent.mkdir(parents=True,exist_ok=True)
pq.write_table(pa.Table.from_pylist([{"sample_id":x,"disease_risk":row_d,"death_risk":row_h} for x in ids],schema=schema),a.output,compression="zstd")
