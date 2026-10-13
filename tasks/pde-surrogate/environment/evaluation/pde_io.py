"""Readers for the three HDF5 layouts used by the task (PBDL corpus, The Well, PDEArena).

read_window(family, path, trajectory, start, n_in, n_out) -> (frames, meta)
    frames: float32 array (n_in + n_out, C, X, Y) in physical units
    meta:   dict with pde, fields, field_scheme, dt, domain_extent, boundary_conditions, constants
family is "ape2d", "well" or "pdearena". The evaluation case builder and the
starter share this module; a delivered package must bundle its own copy.
"""
import json
from pathlib import Path

import h5py
import numpy as np


def read_pbdl(path: Path, traj: int, start: int, n_in: int, n_out: int):
    with h5py.File(path, "r") as f:
        g = f["sims"]
        sim = g[f"sim{traj}"]
        frames = sim[start:start + n_in + n_out].astype(np.float32)
        a = g.attrs
        fields = [str(x) for x in a["Fields"]]
        constants = {str(k): float(sim.attrs[k]) for k in a.get("Constants", []) if k in sim.attrs}
        # the reduced release stores fewer attributes than the full one; fall back to the
        # sidecar json ('all' section, full release) and finally to the corpus defaults
        side = {}
        for cand in (path.with_suffix(".json"), path.parent / "meta_all.json"):
            if cand.exists():
                try:
                    side = json.loads(cand.read_text()).get("all", json.loads(cand.read_text()))
                except Exception:  # noqa: BLE001
                    side = {}
                break
        # per-simulation domain extents (ks, kdv) live in the simulation's constants and take precedence
        if "Domain Extent" in sim.attrs:
            extent = sim.attrs["Domain Extent"]
        elif "Domain Extent" in a:
            extent = a["Domain Extent"]
        else:
            extent = side.get("Domain Extent", 1.0)
        extent = [float(extent)] * 2 if np.ndim(extent) == 0 else [float(x) for x in extent]
        bcs = a["Boundary Conditions"] if "Boundary Conditions" in a else side.get("Boundary Conditions", ["periodic"] * 4)
        meta = {"pde": str(a["PDE"]), "fields": fields, "field_scheme": str(a["Fields Scheme"]),
                "dt": float(a["Dt"]), "domain_extent": extent,
                "boundary_conditions": [str(x) for x in bcs],
                "constants": constants}
    return frames, meta


WELL_BC = {"PERIODIC": "periodic", "WALL": "wall", "OPEN": "open"}


def read_well(path: Path, traj: int, start: int, n_in: int, n_out: int):
    sl = slice(start, start + n_in + n_out)
    chans, names, scheme = [], [], ""
    with h5py.File(path, "r") as f:
        for order, group in enumerate(["t0_fields", "t1_fields", "t2_fields"]):
            for name in f[group].attrs["field_names"]:
                arr = f[group][name][traj, sl].astype(np.float32)  # (T, X, Y[, 2[, 2]])
                if order == 0:
                    chans.append(arr[:, None]); names.append(str(name)); scheme += chr(ord("a") + len(names) - 1)
                else:
                    flat = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], -1)
                    letter = chr(ord("A") + order)  # 'B' for vectors, 'C' for rank-2 tensors
                    for k in range(flat.shape[-1]):
                        chans.append(flat[..., k][:, None]); names.append(f"{name}_{k}"); scheme += letter
                    # distinct letters per field: bump so two consecutive vector fields do not merge
                    scheme = scheme[: -flat.shape[-1]] + chr(ord(letter) + 0) * flat.shape[-1]
                    letter_shift = names.count  # noqa: F841 (documentation only)
        frames = np.concatenate(chans, axis=1)
        t = f["dimensions"]["time"][()]
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
        x, y = f["dimensions"]["x"][()], f["dimensions"]["y"][()]
        extent = [float(x[-1] - x[0] + (x[1] - x[0])), float(y[-1] - y[0] + (y[1] - y[0]))]
        bcs = {}
        for bname in f["boundary_conditions"]:
            g = f["boundary_conditions"][bname]
            for dim in g.attrs["associated_dims"]:
                bcs[str(dim)] = WELL_BC.get(str(g.attrs["bc_type"]), str(g.attrs["bc_type"]).lower())
        constants = {str(k): float(f["scalars"][k][()]) for k in f["scalars"].attrs["field_names"]
                     if not f["scalars"][k].attrs.get("sample_varying", False)}
        constants.update({str(k): float(f.attrs[k]) for k in f.attrs.get("simulation_parameters", [])
                          if k in f.attrs})
        pde = str(f.attrs.get("dataset_name", path.stem))
    # make each vector/tensor field its own run of letters: re-letter groups sequentially
    scheme = _reletter(names)
    meta = {"pde": pde, "fields": names, "field_scheme": scheme, "dt": dt, "domain_extent": extent,
            "boundary_conditions": [bcs.get("x", "periodic")] * 2 + [bcs.get("y", "periodic")] * 2,
            "constants": constants}
    return frames, meta


def read_pdearena(path: Path, traj: int, start: int, n_in: int, n_out: int):
    sl = slice(start, start + n_in + n_out)
    with h5py.File(path, "r") as f:
        g = f[next(iter(f.keys()))]
        frames = np.stack([g[k][traj, sl].astype(np.float32) for k in ("u", "vx", "vy")], axis=1)
        t = g["t"][traj]
        dt = float(t[1] - t[0]) if len(t) > 1 else float(g["dt"][traj])
        dx, dy = float(g["dx"][traj]), float(g["dy"][traj])
        extent = [float(g["u"].shape[2] * dx), float(g["u"].shape[3] * dy)]
        constants = {"buoyancy_y": float(g["buo_y"][traj])}
    meta = {"pde": "Navier-Stokes: buoyant smoke (PDEArena)", "fields": ["u", "vx", "vy"], "field_scheme": "abb",
            "dt": dt, "domain_extent": extent, "boundary_conditions": ["wall"] * 4,  # closed domain: Dirichlet v=0, Neumann scalar (PDEArena paper, App. on NS data)
            "constants": constants}
    return frames, meta


def _reletter(names):
    """Channels named f_0, f_1, ... belong to one field; give each field its own letter."""
    scheme, letter, prev = "", ord("a") - 1, None
    for n in names:
        base = n.rsplit("_", 1)[0] if n.rsplit("_", 1)[-1].isdigit() else n
        if base != prev:
            letter += 1; prev = base
        scheme += chr(letter)
    return scheme


READERS = {"ape2d": read_pbdl, "well": read_well, "pdearena": read_pdearena}


def family_of(source_id: str) -> str:
    return "ape2d" if source_id.startswith("ape2d") else ("pdearena" if source_id.startswith("pdearena") else "well")


def read_window(family, path, trajectory, start, n_in, n_out):
    return READERS[family](Path(path), int(trajectory), int(start), int(n_in), int(n_out))


def trajectory_length(family, path, trajectory=0):
    with h5py.File(path, "r") as f:
        if family == "ape2d":
            return int(f["sims"][f"sim{trajectory}"].shape[0])
        if family == "pdearena":
            return int(f[next(iter(f.keys()))]["u"].shape[1])
        return int(f["dimensions"]["time"].shape[0])
