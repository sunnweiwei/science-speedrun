#!/usr/bin/env python3
"""Build a pde-surrogate-rollout/v1 input directory and its hidden truth directory.

    python3 make_cases.py --data DATA_DIR --manifest split.json --split test \
        --input INPUT_DIR --truth TRUTH_DIR [--context-frames 1]

The manifest lists cases as
  {"case_id": ..., "dataset": "ape2d/burgers", "source": "ape2d", "file": "burgers/burgers.hdf5",
   "trajectory": 57, "start": 0, "n_steps_out": 29, "split": "test"}
for the PBDL layout, and with "file": "data/test/<name>.hdf5" for The Well.
Frames [start, start+context) become `input`; the next n_steps_out become `target`.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from pde_io import family_of, read_window

CONTRACT = "pde-surrogate-rollout/v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True); ap.add_argument("--input", required=True)
    ap.add_argument("--truth", required=True); ap.add_argument("--context-frames", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    cases = [c for c in man["cases"] if c["split"] == args.split]
    if args.limit:
        cases = cases[: args.limit]
    inp, tru = Path(args.input), Path(args.truth)
    (inp / "cases").mkdir(parents=True, exist_ok=True); (tru / "cases").mkdir(parents=True, exist_ok=True)
    index = []
    for c in cases:
        path = Path(args.data) / c["source"] / c["file"]
        frames, meta = read_window(family_of(c["source"]), path, c["trajectory"], c["start"], args.context_frames, c["n_steps_out"])
        if frames.shape[0] != args.context_frames + c["n_steps_out"]:
            raise SystemExit(f"{c['case_id']}: trajectory too short for start={c['start']}")
        x, y = frames[: args.context_frames], frames[args.context_frames:]
        with h5py.File(inp / "cases" / f"{c['case_id']}.h5", "w") as f:
            f.create_dataset("input", data=x)
            for k, v in meta.items():
                f.attrs[k] = json.dumps(v) if k == "constants" else v
            f.attrs["dataset"] = c["dataset"]; f.attrs["n_steps_out"] = c["n_steps_out"]
        with h5py.File(tru / "cases" / f"{c['case_id']}.h5", "w") as f:
            f.create_dataset("target", data=y)
            f.attrs["fields"] = meta["fields"]; f.attrs["field_scheme"] = meta["field_scheme"]
            f.attrs["dataset"] = c["dataset"]
        index.append({"case_id": c["case_id"], "dataset": c["dataset"], "n_steps_out": c["n_steps_out"]})
    (inp / "index.json").write_text(json.dumps({"contract": CONTRACT, "cases": index}, indent=1) + "\n")
    print(f"wrote {len(index)} cases to {inp} and {tru}")


if __name__ == "__main__":
    main()
