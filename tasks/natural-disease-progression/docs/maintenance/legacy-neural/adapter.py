"""Legacy model adapter retained only for historical reproduction."""

from collections import Counter, deque
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq


def fit_adapter(train_dir: Path, max_codes: int = 12000, min_lab_count: int = 100) -> dict:
    db = duckdb.connect()
    events = (train_dir / "events.parquet").as_posix()
    rows = db.execute(f"""SELECT code,unit,count(*) n,
      approx_quantile(TRY_CAST("value" AS DOUBLE),[.25,.5,.75]) quartiles
      FROM read_parquet('{events}') WHERE event_type='lab'
      GROUP BY code,unit HAVING count(*)>={int(min_lab_count)} ORDER BY n DESC LIMIT 512""").fetchall()
    lab_bins = {f"{code}|{unit or ''}": q for code,unit,_,q in rows
                if q and all(value is not None for value in q)}
    freq = Counter()
    for code, kind, n in db.execute(f"""SELECT code,event_type,count(*) FROM read_parquet('{events}')
      GROUP BY code,event_type""").fetchall():
        freq[f"{kind}:{code}"] += n
    for key in lab_bins:
        for bucket in range(4):
            freq[f"lab_value:{key}:q{bucket}"] += 1
        freq[f"lab_value:{key}:missing_or_unfitted"] += 1
    for gender in ("M", "F", "unknown"):
        freq[f"gender:{gender}"] += 1
    for decade in range(0, 100, 10):
        freq[f"age_decade:{decade}"] += 1
    protected = sorted(token for token in freq
                       if token.startswith(("lab_value:", "gender:", "age_decade:")))
    ordered = protected + [token for token in sorted(freq, key=lambda token: (-freq[token], token))
                           if token not in protected]
    vocab = {token: index + 1 for index, token in enumerate(ordered[:max_codes])}
    db.close()
    return {"vocab": vocab, "lab_bins": lab_bins, "max_events": 512}


def event_tokens(event: dict, adapter: dict) -> list[str]:
    kind, code = event["event_type"], event["code"]
    tokens = [f"{kind}:{code}"]
    if kind == "lab":
        unit = event.get("unit") or ""
        key = f"{code}|{unit}"
        quartiles = adapter["lab_bins"].get(key)
        try:
            value = float(event.get("value"))
            finite = np.isfinite(value)
        except (ValueError, TypeError):
            finite = False
        if quartiles and finite:
            bucket = int(np.searchsorted(quartiles, value, side="right"))
            tokens.append(f"lab_value:{key}:q{bucket}")
        else:
            tokens.append(f"lab_value:{key}:missing_or_unfitted")
    return tokens


def history_to_ids(history: dict, adapter: dict, modalities: str = "all") -> list[int]:
    vocab = adapter["vocab"]
    age = history.get("approx_age")
    tokens = [f"gender:{history.get('gender','unknown')}",
              f"age_decade:{min(90, max(0, int(age or 0)//10*10))}"]
    for event in history.get("events", [])[-adapter["max_events"]:]:
        if modalities == "diagnosis" and event["event_type"] != "diagnosis":
            continue
        tokens.extend(event_tokens(event, adapter))
    return [vocab[token] for token in tokens if token in vocab] or [0]


def iter_histories(input_dir: Path, max_events: int = 512):
    samples = sorted(pq.read_table(input_dir / "samples.parquet").to_pylist(),
                     key=lambda row: row["sample_id"])
    db = duckdb.connect()
    path = (input_dir / "events.parquet").as_posix()
    cursor = db.execute(f"""SELECT sample_id,event_type,code,"value",unit,available_time,event_time
      FROM read_parquet('{path}') ORDER BY sample_id,available_time,event_time,event_type,code""")

    def rows():
        while True:
            batch = cursor.fetchmany(65536)
            if not batch:
                break
            yield from batch

    stream = iter(rows())
    row = next(stream, None)
    for sample in samples:
        sid = sample["sample_id"]
        events = deque(maxlen=max_events)
        prior_diagnoses = set()
        while row is not None and row[0] == sid:
            event = dict(zip(("sample_id", "event_type", "code", "value", "unit",
                              "available_time", "event_time"), row))
            if event["event_type"] == "diagnosis" and not event["code"].startswith("ICD9:"):
                prior_diagnoses.add(event["code"])
            events.append(event)
            row = next(stream, None)
        yield sid, {"gender": sample["gender"], "approx_age": sample["approx_age"],
                    "events": list(events), "prior_diagnoses": sorted(prior_diagnoses)}
    db.close()
