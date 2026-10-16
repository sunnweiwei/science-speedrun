"""Build the five off-equilibrium suites added in v2 from pinned public sources.

    python extra_suites.py --cache CACHE --dev DEV_ROOT --private PRIVATE_ROOT

Every source is downloaded once into CACHE and checked against its SHA-256. Splits are
deterministic (SHA-256 of a versioned key). The development tree receives validation inputs and
references; the private tree receives test inputs and references (verifier only).

| suite     | source                                            | val / test                    |
|-----------|---------------------------------------------------|-------------------------------|
| kappa     | PhononDB PBE 103 (Matbench Discovery)             | 40 / 63 materials (stratified)|
| mof       | MOFSimBench PBE-D3 bulk moduli, 100 MOFs          | 40 / 60 MOFs                  |
| amorphous | AM26 (a-C, a-Si, a-SiO2, a-LiPS, a-GST)           | 40 % / 60 % of structures     |
| cleavage  | Cleavage-energy benchmark, 36,718 PBE slabs       | 2,000 / 2,000 slabs, disjoint bulks |
| neb       | LiTraj nebDFT2k, paths without +U elements        | 40 % / 60 % of materials      |
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests

VERSION = "extra-suites-v1"
MBD = "https://ndownloader.figshare.com/files"
MOFSIM = "https://raw.githubusercontent.com/AI4ChemS/mofsim-bench/777c42aed8d80107bd32428e1d6c1029173ce967/mof_benchmark"
SOURCES = {
    "kappa_structures": (f"{MBD}/52179965", "8a7340c66233430211c51de23bffcbc490f21de47414f8ce06613553134fcc3a"),
    "kappa_reference": (f"{MBD}/68221426", "d11d53243fd1586ce5d8bd68f073f94b04d719edb10c3d3f67591f919f86aa2f"),
    "mof_reference": (f"{MOFSIM}/analysis/dft_data/1st_2nd_final_filter_good_bulk_modulus_calc_info_combined.csv",
                      "e72b31750d955fad5a33238ac111ea6467e4eb055b06efb6bf0bac9b85752d99"),
    "am26": ("https://raw.githubusercontent.com/vldgroup/AM26/e88472a56614eaa24d9b4b839f2543249baf6674/data/am26.extxyz",
             "a071f537305e78e3bb1acf5dbffaa9acf1f79e2eb5081cb59218beddaf5a802a"),
    "cleavage": ("https://zenodo.org/records/16970768/files/cleavage_energy_benchmark_DFT_uMLIPs.json?download=1",
                 "bd9e73b677efa3be6053ca5f1b8c5c50bcec185572cd4fe399616437ec3337e0"),
    "nebdft2k": ("https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj/nebDFT2k.zip",
                 "b7a99d89337902e9e1da319f57547170fdb132bf15bf6ffef03a0140e2207d7f"),
}


def h(key: str) -> str:
    return hashlib.sha256(f"{VERSION}:{key}".encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(cache: Path, name: str, url: str | None = None, expected: str | None = None) -> Path:
    url, expected = (url, expected) if url else SOURCES[name]
    path = cache / name
    if not path.exists() or (expected and sha256(path) != expected):
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with requests.get(url, stream=True, timeout=(30, 600), headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            r.raw.decode_content = True  # GitHub raw may gzip text transfers; store the file itself
            with tmp.open("wb") as out:
                shutil.copyfileobj(r.raw, out, 1 << 24)
        tmp.rename(path)
    if expected and sha256(path) != expected:
        raise ValueError(f"checksum mismatch for {name}")
    return path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def out_dirs(dev: Path, private: Path, suite: str):
    return dev / "extra" / suite / "val", private / "extra" / suite / "test"


# ------------------------------------------------------------------------------------- suites
def build_kappa(cache, dev, private):
    from ase.io import read, write
    structures = read(fetch(cache, "kappa_structures"), index=":", format="extxyz")
    ref = {r["material_id"]: r for r in json.load(gzip.open(fetch(cache, "kappa_reference")))}
    ids = sorted(ref)
    logk = {i: np.log10(float(np.ravel(ref[i]["kappa_tot_avg"])[0])) for i in ids}
    q = np.quantile(list(logk.values()), [0.25, 0.5, 0.75])
    groups = {}
    for i in ids:
        spg = int(ref[i]["init_spg_num"])
        groups.setdefault((int(np.searchsorted(q, logk[i])), spg if spg in (186, 216, 225) else 0), []).append(i)
    val = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda i: hashlib.sha256(f"kappa-split-v1:{i}".encode()).hexdigest())
        val += members[: round(len(members) * 40 / 103)]
    for i in sorted(ids, key=lambda i: hashlib.sha256(f"kappa-split-v1:{i}".encode()).hexdigest()):
        if len(val) >= 40:
            break
        if i not in val:
            val.append(i)
    val = set(val[:40])
    for split_ids, root in ((val, out_dirs(dev, private, "kappa")[0]), (set(ids) - val, out_dirs(dev, private, "kappa")[1])):
        root.mkdir(parents=True, exist_ok=True)
        write(root / "inputs.extxyz", [s for s in structures if s.info["material_id"] in split_ids], format="extxyz")
        with gzip.open(root / "references.json.gz", "wt") as f:
            json.dump([ref[i] for i in sorted(split_ids)], f)
        write_json(root / "manifest.json", {"suite": "kappa", "materials": len(split_ids), "ids": sorted(split_ids)})
    return {"val": len(val), "test": len(ids) - len(val)}


def build_mof(cache, dev, private):
    rows = list(csv.DictReader(open(fetch(cache, "mof_reference"))))
    order = sorted(rows, key=lambda r: h("mof:" + r["structure"]))
    split = {"val": order[:40], "test": order[40:]}
    for name, root in zip(("val", "test"), out_dirs(dev, private, "mof")):
        (root / "structures").mkdir(parents=True, exist_ok=True)
        refs, files = [], {}
        for r in split[name]:
            for ext in ("cif", "xyz"):  # 99 MOFs are CIF files, IRMOF-10 is an extended-XYZ file
                try:
                    src = fetch(cache / "mof_structures", f"{r['structure']}.{ext}",
                                f"{MOFSIM}/experiments/structures/main_set/{r['structure']}.{ext}")
                    break
                except requests.HTTPError:
                    continue
            else:
                raise FileNotFoundError(r["structure"])
            files[r["structure"]] = src.name
            shutil.copyfile(src, root / "structures" / src.name)
            refs.append({"id": r["structure"], "file": src.name, "B0_GPa": float(r["B0_GPa"])})
        write_json(root / "references.json", [{"id": x["id"], "B0_GPa": x["B0_GPa"]} for x in refs])
        write_json(root / "manifest.json", {"suite": "mof", "mofs": len(refs),
                                            "files": {x["id"]: x["file"] for x in refs},
                                            "sha256": {x["id"]: sha256(root / "structures" / x["file"]) for x in refs}})
    return {k: len(v) for k, v in split.items()}


def build_amorphous(cache, dev, private):
    from ase.io import read, write
    frames = read(fetch(cache, "am26"), index=":", format="extxyz")
    counts = {}
    for name, root in zip(("val", "test"), out_dirs(dev, private, "amorphous")):
        root.mkdir(parents=True, exist_ok=True)
        keep, refs = [], []
        for i, a in enumerate(frames):
            sid = f"{a.info['system']}-{i:04d}"
            is_val = int(h("am26:" + sid), 16) % 100 < 40
            if is_val != (name == "val"):
                continue
            b = a.copy(); b.calc = None
            b.info = {"id": sid, "system": a.info["system"]}
            keep.append(b)
            refs.append({"id": sid, "system": a.info["system"], "energy_per_atom": float(a.info["energy_per_at"]),
                         "forces": np.asarray(a.get_forces()).round(6).tolist()})
        write(root / "inputs.extxyz", keep, format="extxyz")
        with gzip.open(root / "references.json.gz", "wt") as f:
            json.dump(refs, f)
        write_json(root / "manifest.json", {"suite": "amorphous", "structures": len(refs)})
        counts[name] = len(refs)
    return counts


def build_cleavage(cache, dev, private, per_split=2000):
    data = json.load(open(fetch(cache, "cleavage")))["data"]  # list of records
    rows = [{k: rec[k] for k in ("mpid", "material_id", "miller", "term", "area_slab", "structure_slab",
                                 "structure_bulk", "cleavage_energy_DFT")} for rec in data]
    materials = sorted({r["material_id"] for r in rows}, key=lambda m: h("cleavage-material:" + m))
    val_mat = set(materials[: len(materials) // 2])
    counts = {}
    for name, root in zip(("val", "test"), out_dirs(dev, private, "cleavage")):
        pool = [r for r in rows if (r["material_id"] in val_mat) == (name == "val")]
        pool = sorted(pool, key=lambda r: h(f"cleavage:{r['mpid']}|{r['miller']}|{r['term']}"))[:per_split]
        inputs, refs = [], []
        for r in pool:
            key = f"{r['mpid']}|{r['miller']}|{r['term']}"
            parse = lambda s: s if isinstance(s, dict) else (json.loads(s) if s.lstrip().startswith('{"') else ast.literal_eval(s))
            inputs.append({"id": key, "material_id": r["material_id"], "area": float(r["area_slab"]),
                           "slab": parse(r["structure_slab"]), "bulk": parse(r["structure_bulk"])})
            refs.append({"id": key, "cleavage_energy": float(r["cleavage_energy_DFT"])})
        root.mkdir(parents=True, exist_ok=True)
        with gzip.open(root / "inputs.json.gz", "wt") as f:
            json.dump(inputs, f)
        write_json(root / "references.json", refs)
        write_json(root / "manifest.json", {"suite": "cleavage", "slabs": len(refs),
                                            "materials": sorted({r["material_id"] for r in inputs})})
        counts[name] = len(refs)
    return counts


def build_neb(cache, dev, private):
    z = zipfile.ZipFile(fetch(cache, "nebdft2k"))
    index = list(csv.DictReader(io.TextIOWrapper(z.open("nebDFT2k/nebDFT2k_index.csv"))))
    index = [r for r in index if r["has_specific_TM"] == "False"]
    counts = {}
    for name, root in zip(("val", "test"), out_dirs(dev, private, "neb")):
        rows = [r for r in index if (int(h("neb-material:" + r["material_id"]), 16) % 100 < 40) == (name == "val")]
        (root / "paths").mkdir(parents=True, exist_ok=True)
        for r in rows:
            (root / "paths" / f"{r['edge_id']}_init.xyz").write_bytes(z.read(f"nebDFT2k/{r['edge_id']}_init.xyz"))
        write_json(root / "references.json", [{"id": r["edge_id"], "material_id": r["material_id"],
                                               "barrier_eV": float(r["em_dft"])} for r in rows])
        write_json(root / "manifest.json", {"suite": "neb", "paths": len(rows), "ids": [r["edge_id"] for r in rows]})
        counts[name] = len(rows)
    return counts


BUILDERS = {"kappa": build_kappa, "mof": build_mof, "amorphous": build_amorphous,
            "cleavage": build_cleavage, "neb": build_neb}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument("--suites", default=",".join(BUILDERS))
    a = ap.parse_args()
    path = a.dev / "extra" / "summary.json"
    summary = json.loads(path.read_text())["suites"] if path.exists() else {}
    summary.update({s: BUILDERS[s](a.cache, a.dev, a.private) for s in a.suites.split(",")})
    write_json(path, {"version": VERSION, "suites": summary})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
