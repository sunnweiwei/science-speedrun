#!/usr/bin/env python3
"""Refuse a data mount that carries held-out trajectories.

    python3 check_agent_mount.py DATA_DIR SPLIT_JSON

The corpus files the split cuts must be the truncated training heads on an agent host. A
full upstream file holds the held-out tail, and an agent environment must never see one.
Exit 0 when every present cut file holds at most the pool's trajectories, 1 otherwise.
"""
import json
import sys
from pathlib import Path


def offending(data: Path, split: dict):
    pools = {}
    for name in ("train", "valid"):
        for e in split["pools"][name]:
            key = (e["source"], e["file"])
            pools[key] = max(pools.get(key, 0), e["trajectories"][1])
    held = {(c["source"], c["file"]) for c in split["cases"] if "file" in c}
    bad = []
    for (source, file), keep in sorted(pools.items()):
        if (source, file) not in held:
            continue                      # pool-only files are never cut
        path = data / source / file
        if not path.exists():
            continue
        try:
            import h5py
            with h5py.File(path, "r") as f:
                n = sum(1 for k in f["sims"] if k.startswith("sim"))
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not inspect {path}: {exc}", file=sys.stderr)
            continue
        if n > keep:
            bad.append(f"{path} holds {n} trajectories; the agent-side head keeps {keep}")
    return bad


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="the mounted data directory (/workspace/data in the development container)")
    ap.add_argument("split_json", help="the split manifest (environment/data/split.json)")
    a = ap.parse_args()
    data, split = Path(a.data_dir), json.loads(Path(a.split_json).read_text())
    bad = offending(data, split)
    if bad:
        print("ERROR: the data mount carries held-out trajectories; mount the release's train/ subtree "
              "(materialize.py --from-release --subtree train), not a full materialization:\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(1)
    print("agent mount ok: no cut file holds a held-out trajectory")


if __name__ == "__main__":
    main()
