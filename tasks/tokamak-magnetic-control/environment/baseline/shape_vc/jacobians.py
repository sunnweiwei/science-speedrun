"""Static virtual-circuit Jacobians d(boundary rays, X-points, gaps)/dI per distinct reference shape.

python jacobians.py MODEL.pkl [CACHE_DIR]   (~1 min per reference shape)
"""
import pickle, sys, time
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controller as K
t = time.time()
m = K.build_model(["mastu_diverted_hold", "mastu_high_ip_hold", "mastu_p4_shape_hold", "mastu_ip_ramp",
                   "mastu_pressure_drop", "mastu_sequence", "mastu_vertical_shift"], cache_dir=sys.argv[2] if len(sys.argv) > 2 else None)
pickle.dump(m, open(sys.argv[1], "wb"))
for r in m["refs"]:
    J = r["J"]
    print(r["case"], "J rho norm per coil (mm/kA):", np.round(1e6 * np.sqrt((J[:64] ** 2).mean(0)), 2), "nan", int(np.isnan(J).sum()))
    print("  v0 xlow", r["v0"][64:66], "xup", r["v0"][66:68], "rho range", np.nanmin(r["v0"][:64]).round(3), np.nanmax(r["v0"][:64]).round(3), "zcur", r["zcur"])
print("time", time.time() - t)
