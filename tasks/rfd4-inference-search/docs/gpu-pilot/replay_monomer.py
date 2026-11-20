"""GPU conformance check for NVIDIA's hash-pinned, unconditional 250-residue gallery example."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from biotite.structure.io.pdbx import CIFFile
from diffusers import RFD4ProteinaPipeline
from rfd4_proteina_prep import prepare_cif, write_cif

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "environment/evaluation"))
from score import backbone, scrmsd


def verify_bundle(root):
    for row in (HERE / "fixture.sha256").read_text().splitlines():
        expected, name = row.split(maxsplit=1)
        with (root / name).open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Unexpected fixture content for {name}")


def atom_coordinates(path):
    atoms = CIFFile.read(path).block["atom_site"]
    keys = list(zip(*(atoms[k].as_array(str) for k in ("label_asym_id", "label_seq_id", "label_comp_id", "label_atom_id"))))
    if len(keys) != len(set(keys)):
        raise ValueError("Ambiguous atom correspondence")
    xyz = np.stack([atoms["Cartn_" + dim].as_array(float) for dim in "xyz"], axis=1)
    return dict(zip(keys, xyz, strict=True))


def compare(left, right):
    aa, a = backbone(left, 250)
    bb, b = backbone(right, 250)
    matched = sum(i == j for i, j in zip(aa, bb, strict=True))
    result = {"residues": len(aa), "matching_residues": matched, "sequence_identity": matched / len(aa),
              "ca_rmsd_A": scrmsd(a[1::4], b[1::4]), "backbone_rmsd_A": scrmsd(a, b)}
    if aa == bb:
        first, second = atom_coordinates(left), atom_coordinates(right)
        if first.keys() == second.keys():
            u = np.stack(list(first.values()))
            v = np.stack([second[k] for k in first])
            result.update(common_atoms=len(u), all_atom_rmsd_A=scrmsd(u, v),
                          same_serialized_coordinates=bool(np.array_equal(u, v)))
    return result


def run(bundle, output):
    verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.cuda.init()
    pipe = RFD4ProteinaPipeline.from_pretrained(str(bundle), dtype=torch.bfloat16, local_files_only=True).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    loaded = time.perf_counter() - start
    paths = {"official": bundle / "examples/unconditional/design.cif"}
    runs = []
    for name, seed in (("local42", 42), ("repeat42", 42), ("local43", 43)):
        features, atoms = prepare_cif(bundle / "examples/unconditional/input.cif", seed=seed, device=pipe.device)
        required = ("padding_mask", "is_in_context_conditioning", "supports_local_latent", "token_type", "atom37_mask")
        # The published Quickstart excludes coords_nm from extra conditioning on this noise-start case.
        extra = {k: v for k, v in features.items() if k not in (*required, "coords_nm")}
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = pipe(**{k: features[k] for k in required}, conditioning_features=extra,
                      generator=torch.Generator("cuda:0").manual_seed(seed))
        torch.cuda.synchronize()
        runs.append({"id": name, "seed": seed, "sampling_seconds": time.perf_counter() - start})
        paths[name] = output / (name + ".cif")
        write_cif(result, features, atoms, paths[name])
    comparisons = {k: compare(paths[a], paths[b]) for k, a, b in (
        ("official_to_local", "official", "local42"),
        ("local_repeat", "local42", "repeat42"), ("seed_control", "local42", "local43"))}
    report = {"model_revision": "92fc3ae50e336b2ec59c713c19cfd43e5d3e923b", "gpu": torch.cuda.get_device_name(),
              "torch": str(torch.__version__), "diffusers": version("diffusers"), "dtype": "bfloat16",
              "loading_seconds": loaded, "runs": runs, "comparisons": comparisons}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(comparisons, allow_nan=False))
    if not comparisons["local_repeat"].get("same_serialized_coordinates"):
        raise RuntimeError("Same-device deterministic control failed. See report.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path, help="New directory for local outputs and report")
    args = parser.parse_args()
    run(args.bundle.resolve(), args.output.resolve())
