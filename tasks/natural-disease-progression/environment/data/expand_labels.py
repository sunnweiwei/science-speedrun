"""Write explicit per-horizon labels and masks from compact target states."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from protocol import HORIZONS


def expand(output: Path) -> None:
    db = duckdb.connect()
    for section in (Path("train"), Path("trusted/validation"), Path("trusted/test")):
        source = (output / section / "labels.parquet").as_posix()
        target = (output / section / "label_windows.parquet").as_posix()
        cols = ["sample_id", "disease_code", "risk_set", "ambiguous_prior",
                "first_future_record_time", "future_same_time_group", "event_days",
                "cutoff", "diagnosis_observation_end", "death_days", "death_observation_end"]
        for horizon in HORIZONS:
            positive = (f"event_days BETWEEN 0 AND {horizon} AND "
                        "cutoff+event_days*INTERVAL 1 DAY<=diagnosis_observation_end AND "
                        "(death_days IS NULL OR event_days<=death_days)")
            eligible = "risk_set AND NOT ambiguous_prior"
            observed = (f"cutoff+INTERVAL {horizon} DAY<=diagnosis_observation_end OR "
                        f"(death_days BETWEEN 0 AND {horizon} AND "
                        "cutoff+death_days*INTERVAL 1 DAY<=diagnosis_observation_end)")
            cols.append(f"CAST(({eligible}) AND COALESCE(({positive}),FALSE) AS BOOLEAN) AS disease_label_{horizon}")
            cols.append(f"CAST(({eligible}) AND (COALESCE(({positive}),FALSE) OR COALESCE(({observed}),FALSE)) AS BOOLEAN) AS disease_valid_{horizon}")
            cols.append(f"CAST(COALESCE(death_days BETWEEN 0 AND {horizon},FALSE) AS BOOLEAN) AS death_label_{horizon}")
            cols.append(f"CAST(COALESCE((death_days BETWEEN 0 AND {horizon}),FALSE) OR cutoff+INTERVAL {horizon} DAY<=death_observation_end AS BOOLEAN) AS death_valid_{horizon}")
        statement = (f"SELECT {','.join(cols)} FROM read_parquet('{source}') "
                     "ORDER BY sample_id,disease_code")
        db.execute(f"COPY ({statement}) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    expand(parser.parse_args().output)
