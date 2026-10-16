"""Drop frames of the original 500k MP fine-tuning set that touch a v2 evaluation material.

    python filter_standard.py --standard STANDARD_V1_DATA --blocked POOL/blocked.json --output DIR

The original task sampled 500,000 + 5,000 MPtrj/sAlex frames (`prepare_standard.py`) while
excluding its own three suites. v2 adds five suites; about 0.2 % of those frames share a
composition or parent material with them. The output keeps the original order and format, with a
new manifest, so `baseline/uma-mp-finetune/prepare_training.py` runs on it unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from common import decode_atoms, digest, write_json  # noqa: E402
from splits import composition_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--standard", type=Path, required=True)
    ap.add_argument("--blocked", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    blocked = json.loads(a.blocked.read_text())
    comps, ids = set(blocked["compositions"]), set(blocked["material_ids"])
    manifest = json.loads((a.standard / "manifest.json").read_text())
    a.output.mkdir(parents=True)
    dropped = {}
    for split in ("train", "val"):
        kept = removed = 0
        with (a.standard / f"{split}.jsonl").open() as src, (a.output / f"{split}.jsonl").open("w") as dst:
            for line in src:
                row = json.loads(line)
                comp = composition_key(Counter(decode_atoms(row["atoms"]).get_chemical_symbols()))
                if comp in comps or row["parent"] in ids:
                    removed += 1
                    continue
                dst.write(line)
                kept += 1
        dropped[split] = {"kept": kept, "removed": removed}
    shutil.copyfile(a.standard / "element-references.yaml", a.output / "element-references.yaml")
    manifest.update(
        version=manifest.get("version", "standard-v1") + "+v2-exclusions",
        files={f"{s}.jsonl": digest(a.output / f"{s}.jsonl") for s in ("train", "val")},
        v2_exclusions={"blocked_sha256": digest(a.blocked), "frames": dropped,
                       "parent_manifest_sha256": digest(a.standard / "manifest.json")},
    )
    write_json(a.output / "manifest.json", manifest)
    print(json.dumps(dropped))


if __name__ == "__main__":
    main()
