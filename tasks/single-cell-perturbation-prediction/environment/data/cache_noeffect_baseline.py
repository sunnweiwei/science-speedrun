#!/usr/bin/env python3
"""Operator stage: score the no-effect baseline once per test context and subset and cache the
cell-eval tables under DATA_ROOT/private/<ctx>/noeffect/<subset>/ so the grader normalises
against a fixed reference; likewise for every training context's dev_split (shipped to the
agent under dev_split/noeffect/all/, so run_loco.py only has to score the candidate side).
Re-run whenever cell-eval or the data change."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))  # environment/evaluation/cellscore.py
from cellscore import (assemble_prediction, assemble_real, cell_eval_version, noeffect_prediction,  # noqa: E402
                       read_requests, run_cell_eval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--skip-dev", action="store_true")
    ap.add_argument("--skip-private", action="store_true")
    a = ap.parse_args()
    if not a.skip_dev:
        for split in sorted(p / "dev_split" for p in (a.root / "dev" / "train").iterdir() if (p / "dev_split" / "responses.h5ad").exists()):
            requests = read_requests(split / "requests.jsonl")
            basal = ad.read_h5ad(split / "basal.h5ad")
            resp = ad.read_h5ad(split / "responses.h5ad")
            base = noeffect_prediction(basal, requests)
            out = split / "noeffect" / "all"
            res, agg = run_cell_eval(assemble_prediction(base, basal, requests, None), assemble_real(resp, basal, requests, None),
                                     out / "cell-eval", a.threads or 32, "all", de_real=out / "real_de.csv")
            res.write_csv(out / "results.csv"); agg.write_csv(out / "agg_results.csv")
            real_de = out / "cell-eval" / "all_real_de.csv"
            if real_de.exists():
                real_de.replace(out / "real_de.csv")
            shutil.rmtree(out / "cell-eval", ignore_errors=True)
            (split / "noeffect" / "manifest.json").write_text(json.dumps({"cell_eval_version": cell_eval_version()}, indent=2) + "\n")
            print(f"[noeffect] dev {split.parent.name}: {agg.to_pandas().set_index('statistic').loc['mean', ['overlap_at_N', 'discrimination_score_l1', 'mae']].to_dict()}")
    if a.skip_private:
        return
    for priv in sorted(p for p in (a.root / "private").iterdir() if (p / "responses.h5ad").exists()):
        ctx = priv.name
        pm = json.loads((priv / "manifest.json").read_text())
        public = a.root / "dev" / "test" / ctx / "public"
        requests = read_requests(public / "requests.jsonl")
        basal = ad.read_h5ad(priv / "basal.h5ad")
        resp = ad.read_h5ad(priv / "responses.h5ad")
        base = noeffect_prediction(basal, requests)
        subsets = {"all": None,
                   "leaderboard": {r["request_id"] for r in pm["requests"] if r["subset"] == "leaderboard"},
                   "final": {r["request_id"] for r in pm["requests"] if r["subset"] == "final"}}
        for tag, keep in subsets.items():
            out = priv / "noeffect" / tag
            res, agg = run_cell_eval(assemble_prediction(base, basal, requests, keep), assemble_real(resp, basal, requests, keep),
                                     out / "cell-eval", a.threads or 32, tag, de_real=out / "real_de.csv")
            res.write_csv(out / "results.csv"); agg.write_csv(out / "agg_results.csv")
            real_de = out / "cell-eval" / f"{tag}_real_de.csv"
            if real_de.exists():
                real_de.replace(out / "real_de.csv")  # reused by the grader for every submission
            shutil.rmtree(out / "cell-eval", ignore_errors=True)
            print(f"[noeffect] {ctx}/{tag}: {agg.to_pandas().set_index('statistic').loc['mean', ['overlap_at_N', 'discrimination_score_l1', 'mae']].to_dict()}")
        (priv / "noeffect" / "manifest.json").write_text(json.dumps({"cell_eval_version": cell_eval_version(),
                                                                       "responses_sha256": pm["responses_sha256"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
