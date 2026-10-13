#!/usr/bin/env python3
"""Write a tiny synthetic example of the contract's INPUT_DIR, the hidden TRUTH_DIR and a valid OUTPUT_DIR.

    python3 make_example.py OUT_DIR      # OUT_DIR/{input,truth,output}; then score.py on them gives a real number
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out_dir", help="directory to create; receives input/, truth/ and output/")
root = Path(ap.parse_args().out_dir); rng = np.random.default_rng(0)
for d in ("input/cases", "truth/cases", "output/cases"):
    (root / d).mkdir(parents=True, exist_ok=True)
cases = []
for i, (ds, C, X, Y, K) in enumerate([("ape2d/burgers", 2, 32, 32, 26), ("well/turbulent_radiative_layer_2D", 4, 16, 48, 30)]):
    base = rng.normal(size=(C, X, Y)).astype(np.float32)
    traj = np.stack([base * (1 + 0.02 * t) + 0.01 * t for t in range(4 + K)])          # (4+K, C, X, Y)
    cid = f"example-{i}"
    with h5py.File(root / "input/cases" / f"{cid}.h5", "w") as f:
        f.create_dataset("input", data=traj[:4])
        f.attrs["dataset"] = ds; f.attrs["pde"] = "Burgers" if i == 0 else "turbulent_radiative_layer_2D"
        f.attrs["fields"] = ["Velocity X", "Velocity Y"] if i == 0 else ["density", "pressure", "velocity_0", "velocity_1"]
        f.attrs["field_scheme"] = "VV" if i == 0 else "abcc"; f.attrs["dt"] = 0.01 if i == 0 else 1.597
        f.attrs["domain_extent"] = [1.0, 1.0] if i == 0 else [1.0, 3.0]
        f.attrs["boundary_conditions"] = ["periodic"] * 4 if i == 0 else ["periodic", "periodic", "open", "open"]
        f.attrs["constants"] = json.dumps({"Viscosity": 1e-4} if i == 0 else {"tcool": 0.03}); f.attrs["n_steps_out"] = K
    with h5py.File(root / "truth/cases" / f"{cid}.h5", "w") as f:
        f.create_dataset("target", data=traj[4:]); f.attrs["fields"] = list(f"{c}" for c in range(C)) if False else (["Velocity X", "Velocity Y"] if i == 0 else ["density", "pressure", "velocity_0", "velocity_1"])
        f.attrs["field_scheme"] = "VV" if i == 0 else "abcc"; f.attrs["dataset"] = ds
    with h5py.File(root / "output/cases" / f"{cid}.h5", "w") as f:               # a valid (persistence) output
        f.create_dataset("prediction", data=np.repeat(traj[3:4], K, axis=0))
    cases.append({"case_id": cid, "dataset": ds, "n_steps_out": K})
(root / "input/index.json").write_text(json.dumps({"contract": "pde-surrogate-rollout/v1", "cases": cases}, indent=1))
print(f"example written under {root}: {len(cases)} cases")
