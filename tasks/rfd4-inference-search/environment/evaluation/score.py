"""Independent AF3 refolding, backbone self-consistency and manuscript S12 clustering."""
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure.io.pdbx import CIFFile

BACKBONE = ("N", "CA", "C", "O")
THRESHOLD = 1.5


def backbone(path, length):
    """Use the same residue/atom order in both structures, independent of CIF row order."""
    atoms = CIFFile.read(path).block["atom_site"]
    def field(k):
        return atoms[k].as_array(str)
    chain, residue, comp, names = map(field, ("label_asym_id", "label_seq_id", "label_comp_id", "label_atom_id"))
    if len(set(chain)) != 1:
        raise ValueError("Expected one chain")
    coord = np.stack([atoms["Cartn_" + k].as_array(float) for k in "xyz"], axis=1)
    xyz = np.full((length, len(BACKBONE), 3), np.nan)
    seq = [""] * length
    for i, name in enumerate(names):
        if name not in BACKBONE:
            continue
        j = int(residue[i]) - 1
        if not 0 <= j < length:
            raise ValueError("Unexpected residue index")
        aa = ProteinSequence.convert_letter_3to1(comp[i])
        if aa not in "ACDEFGHIKLMNPQRSTVWY" or seq[j] not in ("", aa):
            raise ValueError("Noncanonical or inconsistent identity")
        k = BACKBONE.index(name)
        if np.isfinite(xyz[j, k]).any():
            raise ValueError("Duplicate backbone atom")
        seq[j], xyz[j, k] = aa, coord[i]
    if not all(seq) or not np.isfinite(xyz).all():
        raise ValueError("Incomplete backbone")
    return "".join(seq), xyz.reshape(-1, 3)


def scrmsd(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    u[:, -1] *= np.linalg.det(u @ vt)
    return float(np.sqrt(np.mean(np.sum((x @ u @ vt - y) ** 2, axis=1))))


def cluster(ids, folder, foldseek):
    if len(ids) <= 1:
        return {s: s for s in ids}
    with (folder / "foldseek.log").open("w") as log:
        subprocess.run([str(foldseek), "easy-cluster", str(folder / "structures"), str(folder / "clusters"),
                        str(folder / "tmp"), "--alignment-type", "1", "--cov-mode", "0",
                        "--min-seq-id", "0", "--tmscore-threshold", "0.5", "--threads", "4"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    with (folder / "clusters_cluster.tsv").open() as f:
        pairs = [(Path(rep).stem, Path(member).stem) for rep, member in csv.reader(f, delimiter="\t")]
    if len(pairs) != len(ids) or {m for _, m in pairs} != set(ids) or any(r not in ids for r, _ in pairs):
        raise RuntimeError("Incomplete Foldseek cluster assignment")
    return {m: r for r, m in pairs}


def score(outputs, slots, assets, work, af_seed):
    """Every returned metric derives from verifier-owned structures and assignments."""
    work.mkdir(parents=True, exist_ok=False)
    queries = work / "queries"
    queries.mkdir()
    cohorts = {}
    for length, paths in outputs.items():
        folder = work / str(length) / "structures"
        folder.mkdir(parents=True)
        valid = {}
        for i, path in enumerate(paths):
            key = f"l{length}d{i:03d}"
            try:
                seq, coords = backbone(path, length)
            except (ValueError, KeyError):
                continue  # Invalid native samples still occupy their allocated slot.
            valid[key] = (seq, coords)
            (folder / f"{key}.cif").symlink_to(path.resolve())
            query = {"name": key, "modelSeeds": [af_seed], "dialect": "alphafold3", "version": 1,
                     "sequences": [{"protein": {"id": "A", "sequence": seq,
                                   "unpairedMsa": "", "pairedMsa": "", "templates": []}}]}
            (queries / f"{key}.json").write_text(json.dumps(query))
        cohorts[length] = valid

    start = time.monotonic()
    if any(cohorts.values()):
        cmd = [os.environ.get("AF3_PYTHON", "/opt/af3env/bin/python"),
               os.environ.get("AF3_SCRIPT", "/opt/af3/run_alphafold.py"),
               f"--input_dir={queries}", f"--output_dir={work / 'refolds'}", f"--model_dir={assets / 'af3'}",
               "--run_data_pipeline=false", "--num_recycles=5", "--num_diffusion_samples=1",
               "--flash_attention_implementation=xla", "--buckets=256",
               f"--jax_compilation_cache_dir={work / 'jax-cache'}"]
        with (work / "af3.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    af_seconds = time.monotonic() - start
    breakdown = {}
    for length, valid in cohorts.items():
        assignments = cluster(valid.keys(), work / str(length), assets / "foldseek")
        passed = []
        measurements = {}
        for key, (seq, xyz) in valid.items():
            predicted = work / "refolds" / key / f"{key}_model.cif"
            refseq, refxyz = backbone(predicted, length)
            if seq != refseq:
                raise RuntimeError("AF3 refold has a different sequence")
            rmsd = scrmsd(xyz, refxyz)
            measurements[key] = rmsd
            if rmsd <= THRESHOLD:
                passed.append(key)
        unique = len({assignments[k] for k in passed})
        (work / str(length) / "measurements.json").write_text(json.dumps({
            k: {"scRMSD_A": v, "cluster": assignments[k]} for k, v in measurements.items()}, indent=2))
        breakdown[length] = {"slots": slots, "emitted": len(outputs[length]), "refolded": len(valid),
                             "passes": len(passed), "passing_clusters": unique,
                             "pass_rate": len(passed) / slots, "clustered_pass_rate": unique / slots}
    reward = sum(x["clustered_pass_rate"] for x in breakdown.values()) / len(breakdown)
    return {"reward": reward, "by_length": breakdown, "af3_seconds": af_seconds}
