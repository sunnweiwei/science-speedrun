#!/usr/bin/env python3
"""Regenerate sources.lock.json from the Hugging Face API at pinned revisions.

For every file of every pinned source the lock records the path, byte size and
a content digest: the upstream LFS SHA-256 for LFS files, the git blob SHA-1
(sha1("blob <size>\\0" + bytes)) for the small non-LFS files. materialize.py
verifies both, so every byte that reaches the task is checked against a
digest that was recorded from the pinned commit.

Run with network access:  python3 build_sources_lock.py [--out sources.lock.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

GIFTEVAL_REV = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
PRETRAIN_REV = "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874"
LEADERBOARD_REV = "a5019972f79e650b1ede252885df5f515fa115b8"

SOURCES = [
    {"id": "gifteval", "repo": "Salesforce/GiftEval", "repo_type": "dataset", "revision": GIFTEVAL_REV, "license": "Apache-2.0",
     "role": "evaluation benchmark: 23 datasets in 28 directories / 55 dataset-frequency variants / 97 configs; test windows are the last 10% of every series (arXiv:2410.10393)",
     "layout": "datasets.save_to_disk Arrow: <dataset>/[<freq>/]data-00000-of-00001.arrow + dataset_info.json + state.json"},
    {"id": "gifteval_pretrain", "repo": "Salesforce/GiftEvalPretrain", "repo_type": "dataset", "revision": PRETRAIN_REV, "license": "Apache-2.0",
     "role": "pretraining corpus constructed by the benchmark authors to have no series in common with GIFT-Eval; only the allow-listed directories below are part of the task",
     "layout": "datasets.save_to_disk Arrow: <dataset>/data-*-of-*.arrow + dataset_info.json + state.json"},
    {"id": "leaderboard", "repo": "Salesforce/GIFT-Eval", "repo_type": "space", "revision": LEADERBOARD_REV, "license": "see space README",
     "role": "leaderboard snapshot: results/<model>/all_results.csv + config.json for every published system (per-config MASE and WQL); reference only, never mounted in the agent environment"},
]
# GiftEvalPretrain directories excluded from the task: most climate reanalysis / projection shards. Since v2 the
# allow-list keeps buildings_900k, borg_cluster_data_2011, nine ERA5 years (2010-2018) and three CMIP6 shards
# (2000, 2005, 2010), about 233 GB: the TimesFM-3-architecture baseline measured that more pretraining data raises the
# test reward (docs/validation/corpus_scaling.json). A content check found none of their series in GIFT-Eval
# (docs/validation/extra_dirs_leak_check.json).
PRETRAIN_KEEP_CLIMATE = {f"era5_{y}" for y in range(2010, 2019)} | {"cmip6_2000", "cmip6_2005", "cmip6_2010"}
PRETRAIN_EXCLUDE_RE = re.compile(r"^(era5_|cmip6_)")
PRETRAIN_EXCLUDE = set()
# Directories that contain GIFT-Eval evaluation series. The corpus is advertised as non-leaking and no dataset name
# is shared, but the underlying series are: M4 and the Monash archive were assembled from overlapping source pools.
# Verified by content (affine-invariant fingerprints, then Pearson correlation on the raw values), see
# docs/validation/corpus_overlap.json: m4_hourly is pedestrian_counts + 10 (r = 1.000000 over all 748 observations,
# test horizon included), and m4_monthly, m4_yearly and temperature_rain series recur in tourism_monthly, fred_md,
# monash_m3_yearly and weather. beijing_air_quality is excluded as a precaution: KDD Cup 2018 is Beijing air
# quality, and the two share 46 anchor windows though no whole series matched.
PRETRAIN_LEAKING = {"pedestrian_counts", "weather", "tourism_monthly", "fred_md", "monash_m3_yearly",
                    "beijing_air_quality"}
FAST_TIER_MAX_BYTES = 200_000_000  # directories under this size form the ~1.45 GB 'fast' tier for smoke runs

# released checkpoints replayed as the external target (staged only by maintainers via materialize.py --pretrained)
PRETRAINED = [
    {"id": "timesfm-3.0", "repo": "google/timesfm-3.0-pytorch", "revision": "43046b85ec22d584a13f8098c2ed39c889e129c2",
     "license": "timesfm-non-commercial-license-v1.0 (non-commercial; not redistributed)", "leaderboard_dir": "TimesFM-3"},
    {"id": "timesfm-2.5", "repo": "google/timesfm-2.5-200m-pytorch", "revision": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
     "license": "Apache-2.0", "leaderboard_dir": "TimesFM-2.5"},
    {"id": "chronos-2", "repo": "amazon/chronos-2", "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
     "license": "Apache-2.0", "leaderboard_dir": "Chronos-2"},
    {"id": "chronos-bolt-base", "repo": "amazon/chronos-bolt-base", "revision": "5d9f166d69f47aef3401367a7b842e78fe97b121",
     "license": "Apache-2.0", "leaderboard_dir": "chronos_bolt_base"},
]


def list_files(api, src):
    files = []
    for e in api.list_repo_tree(src["repo"], repo_type=src["repo_type"], revision=src["revision"], recursive=True, expand=True):
        if getattr(e, "size", None) is None:
            continue  # directory
        lfs = getattr(e, "lfs", None)
        files.append({"path": e.path, "bytes": int(e.size),
                      "sha256": lfs.get("sha256") if lfs else None, "git_blob_sha1": e.blob_id})
    return sorted(files, key=lambda f: f["path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("sources.lock.json")))
    args = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    lock = {"schema": "timeseries-forecasting-sources-lock/v1", "sources": [], "pretrained": []}
    for src in SOURCES:
        info = (api.dataset_info if src["repo_type"] == "dataset" else api.space_info)(src["repo"], revision=src["revision"])
        assert info.sha == src["revision"], (src["repo"], info.sha)
        files = list_files(api, src)
        entry = {**src, "hf_last_modified": str(info.last_modified)}
        if src["id"] == "gifteval_pretrain":
            dirs = {}
            for f in files:
                top = f["path"].split("/")[0]
                if "/" in f["path"]:
                    d = dirs.setdefault(top, {"bytes": 0, "files": 0})
                    d["bytes"] += f["bytes"]; d["files"] += 1
            allowed = sorted(d for d in dirs if (not PRETRAIN_EXCLUDE_RE.match(d) or d in PRETRAIN_KEEP_CLIMATE)
                             and d not in PRETRAIN_EXCLUDE and d not in PRETRAIN_LEAKING)
            excluded = sorted(d for d in dirs if d not in allowed)
            entry["directories"] = {d: {**dirs[d], "tier": "fast" if dirs[d]["bytes"] < FAST_TIER_MAX_BYTES else "full"} for d in allowed}
            entry["excluded_directories"] = {d: dirs[d] for d in excluded}
            entry["excluded_bytes"] = sum(dirs[d]["bytes"] for d in excluded)
            entry["excluded_for_leakage"] = {d: dirs[d] for d in sorted(PRETRAIN_LEAKING) if d in dirs}
            files = [f for f in files if "/" not in f["path"] or f["path"].split("/")[0] in allowed]
        if src["id"] == "leaderboard":
            files = [f for f in files if f["path"].startswith("results/") or f["path"] in ("README.md", "src/utils.py")]
        entry["files"] = files
        entry["summary"] = {"files": len(files), "bytes": sum(f["bytes"] for f in files),
                            "lfs_files": sum(1 for f in files if f["sha256"]), "non_lfs_files": sum(1 for f in files if not f["sha256"])}
        lock["sources"].append(entry)
        print(f"{src['id']:20s} {entry['summary']}", file=sys.stderr)
    for p in PRETRAINED:
        info = api.model_info(p["repo"], revision=p["revision"], files_metadata=True)
        assert info.sha == p["revision"], (p["repo"], info.sha)
        files = [{"path": s.rfilename, "bytes": int(s.size or 0), "sha256": (s.lfs or {}).get("sha256") if getattr(s, "lfs", None) else None}
                 for s in info.siblings]
        lock["pretrained"].append({**p, "hf_last_modified": str(info.last_modified), "files": files,
                                   "bytes": sum(f["bytes"] for f in files)})
        print(f"{p['id']:20s} {len(files)} files {sum(f['bytes'] for f in files)/1e9:.2f} GB", file=sys.stderr)
    Path(args.out).write_text(json.dumps(lock, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
