"""materialize stages. Usage: python -m materialize.cli STAGE [options]

  verify     check pinned sources exist with the recorded size/sha256
  panel      build the master gene panel into WORK
  harmonize  load one or all sources into WORK (--only NAME)
  build      assemble DATA_ROOT from WORK
  embed      SE-600M embeddings for train contexts (GPU; needs arc-state)
  priors     ESM-2 / GO / DepMap prior files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import pandas as pd

from . import config as cfg
from .build import WORK_PANEL, build, write_intermediate
from .context import write_json
from .genes import build_gene_universe, build_master_panel
from .loaders import load_orion, load_replogle_format, load_vcc2025

HERE = Path(__file__).resolve().parent


def _sources_manifest() -> dict:
    return json.loads((HERE.parent / "download-source.json").read_text())


def stage_verify(cache: Path, strict: bool) -> dict:
    from .build import sha256
    manifest = _sources_manifest()
    out = {}
    for key, entry in manifest["files"].items():
        path = cache / entry["path"]
        ok = path.exists() and (entry.get("size") is None or path.stat().st_size == entry["size"])
        digest = None
        if ok and strict and entry.get("sha256"):
            digest = sha256(path)
            ok = digest == entry["sha256"]
        out[key] = {"path": str(path), "ok": ok, "size": path.stat().st_size if path.exists() else None, "sha256": digest}
        print(f"[verify] {'OK ' if ok else 'BAD'} {key} -> {path}")
    if not all(v["ok"] for v in out.values()):
        raise SystemExit("source verification failed")
    return manifest


def _replogle_var(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        cats = f["var/__categories/gene_name"][:]
        codes = f["var/gene_name"][:]
        return pd.DataFrame({"gene_id": [x.decode() for x in f["var/gene_id"][:]],
                             "gene_name": [cats[c].decode() for c in codes]})


def _load_panel_inputs(cache: Path):
    vcc = [s for s in cfg.SOURCES if s["loader"] == "vcc"][0]
    with h5py.File(cache / vcc["path"][0], "r") as f:
        vcc_var = pd.DataFrame({"gene_id": [x.decode() for x in f["var/gene_id"][:]]},
                               index=[x.decode() for x in f["var/_index"][:]])
    orion_root = cache / [s for s in cfg.SOURCES if s["loader"] == "orion"][0]["path"]
    orion = pd.read_parquet(orion_root / "metadata" / "gene_metadata.parquet")
    rep = {s["name"]: _replogle_var(cache / s["path"]) for s in cfg.SOURCES if s["loader"] == "replogle"}
    return vcc_var, orion, rep


def stage_panel(cache: Path, work: Path) -> None:
    vcc_var, orion, rep = _load_panel_inputs(cache)
    master = build_master_panel(vcc_var, orion, rep)
    work.mkdir(parents=True, exist_ok=True)
    master.to_csv(work / WORK_PANEL)
    universe = build_gene_universe(master, orion, rep)
    write_json(work / "gene_universe.json", universe)
    print(f"[panel] {len(master)} genes ({int(master['in_vcc_panel'].sum())} VCC panel); universe {len(universe)}")


def stage_harmonize(cache: Path, work: Path, only: list[str] | None, workers: int) -> None:
    master = pd.read_csv(work / WORK_PANEL, index_col=0)
    universe = json.loads((work / "gene_universe.json").read_text())
    for s in cfg.SOURCES:
        if only and s["name"] not in only:
            continue
        if (work / s["name"] / "stats.json").exists() and not os.environ.get("MATERIALIZE_FORCE"):
            print(f"[harmonize] {s['name']} exists, skipping"); continue
        t0 = time.time()
        if s["loader"] == "replogle":
            L = load_replogle_format(cache / s["path"], s["name"], master, universe, s["chemistry"], s["source"],
                                     s["keep_per_target"], s["keep_controls"], cfg.SEED)
        elif s["loader"] == "vcc":
            L = load_vcc2025([cache / p for p in s["path"]], s["name"], master, universe,
                             s["keep_per_target"], s["keep_controls"], cfg.SEED)
        elif s["loader"] == "orion":
            L = load_orion(cache / s["path"], s["line"], s["name"], master, universe, s["keep_per_target"],
                           s["keep_controls"], cfg.SEED, workers=workers, **cfg.ORION_QC)
        else:
            raise ValueError(s["loader"])
        write_intermediate(work, s["name"], L.context, L.pseudobulk, {**L.stats, "seconds": round(time.time() - t0, 1)})
        print(f"[harmonize] {s['name']}: {L.stats} in {time.time() - t0:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["verify", "panel", "harmonize", "build", "embed", "priors"])
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--strict", action="store_true", help="verify sha256 (slow)")
    ap.add_argument("--se-dir", type=Path)
    ap.add_argument("--gpus", type=str, default="0")
    a = ap.parse_args(argv)
    if a.stage == "verify":
        stage_verify(a.cache, a.strict)
    elif a.stage == "panel":
        stage_panel(a.cache, a.work)
    elif a.stage == "harmonize":
        stage_harmonize(a.cache, a.work, a.only, a.workers)
    elif a.stage == "build":
        build(a.work, a.root, _sources_manifest())
    elif a.stage == "embed":
        from .embed import stage_embed
        stage_embed(a.root, a.se_dir or (a.cache / "se600m"), a.gpus)
    elif a.stage == "priors":
        from .priors import stage_priors
        stage_priors(a.cache, a.root)


if __name__ == "__main__":
    main()
