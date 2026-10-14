"""Assemble the NumPy-only tokamak-policy/v1 package: python build_package.py --out PKG --est EST.npz --model MODEL.pkl"""
import argparse, json, pickle, shutil, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controller as K

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--est", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--params", default="{}")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
m = pickle.loads(Path(a.model).read_bytes())
arr = {"R": m["R"], "L": m["L"], "lim": m["lim"], "vmax": m["vmax"], "n_refs": np.array(len(m["refs"])),
       "ip": np.array([r["ip"] for r in m["refs"]]), "zcur": np.array([r["zcur"] for r in m["refs"]]),
       "gaps": np.array([r["gaps"] for r in m["refs"]])}
for i, r in enumerate(m["refs"]):
    arr[f"pts{i}"] = r["pts"]; arr[f"J{i}"] = r["J"]
np.savez(out / "model.npz", **arr)
params = dict(K.DEFAULTS, **json.loads(a.params)); params.pop("excite", None)
(out / "params.json").write_text(json.dumps(params, indent=1))
shutil.copy2(a.est, out / "estimator.npz")
shutil.copy2(HERE / "policy.py", out / "policy.py")
(out / "infer.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$(readlink -f \"$0\")\")\"\n# one BLAS thread: the per-step algebra is tiny, and threaded BLAS stalls under a loaded machine\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\nexec python3 policy.py\n")
(out / "infer.sh").chmod(0o755)
print("package", out, "refs", [r["case"] for r in m["refs"]])
