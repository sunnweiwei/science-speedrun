"""Compare public validation config target IDs to the pinned Figure 2 / S29 roster."""
import argparse
import csv
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
REVISION = "6306a210dd17dcae8e4e0f381b5b74a719d9b30e"
CONFIGS = {
    "ppi/ppi_bb_only_subset_12": "ppi",
    "mcsa/mcsa_subset": "ame",
    "small_molecule_binder/disco_subset": "sm_binder",
    "small_molecule_binder/disco_figure_2c": "sm_binder",
    "dna_binder/dna_binder_sequence_context": "dna",
}


def target_id(name, criterion):
    key = name.casefold()
    if criterion == "ppi":
        return key.removeprefix("rfd3_")
    if criterion == "dna":
        return key.removesuffix("_sequence_context")
    return key


def audit(upstream):
    rosters = {}
    with (ROOT / "environment/references/overall/targets.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == "Low":
                criterion = row["criterion"]
                rosters.setdefault(criterion, set()).add(target_id(row["target"], criterion))
    for config, criterion in CONFIGS.items():
        path = f"configs/datasets/val/{config}.yaml"
        raw = subprocess.check_output(["git", "-C", str(upstream), "show", f"{REVISION}:{path}"])
        keys = yaml.safe_load(raw)["dataset"]["subset_to_keys"]
        ids = {target_id(k, criterion) for k in keys}
        if len(ids) != len(keys):
            raise ValueError(f"Duplicate target after ID normalization in {path}")
        yield config, len(ids), len(ids & rosters[criterion]), len(rosters[criterion])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Upstream checkout containing the pinned revision")
    args = parser.parse_args()
    print("Validation config,Listed targets,Matched official targets,Official suite size")
    for result in audit(args.upstream):
        print(*result, sep=",")
