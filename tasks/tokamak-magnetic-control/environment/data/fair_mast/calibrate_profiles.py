"""Map FreeGSNKE profile parameters to beta_p and li_3 around the base case (FIX_PROPOSAL fix 3).

The FAIR-MAST statistics in derived_ranges.json are in beta_p (EFIT definition, FreeGSNKE
poloidalBeta1) and li_3 (FreeGSNKE internalInductance3). The task randomizes ConstrainPaxisIp
parameters, so this script solves static equilibria from the base case's reference currents
one parameter at a time (+/- STEP in ln paxis, alpha_m and alpha_n; large joint perturbations
of a fixed-current equilibrium often stall the solver) and fits the local sensitivities
d(ln beta_p)/d(.) and d(li_3)/d(.).
Needs FreeGSNKE (the task images, or `pip install freegsnke==3.1.0`).

Usage: python environment/data/fair_mast/calibrate_profiles.py [--output environment/data/fair_mast/profile_calibration.json]
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # environment/
from tokagym import TokamakMagneticEnv  # noqa: E402
from tokagym import cases as C  # noqa: E402
from tokagym.env import EquilibriumError  # noqa: E402

STEP = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="mastu_diverted_hold")
    ap.add_argument("--output", default=str(HERE / "profile_calibration.json"))
    args = ap.parse_args()
    env = TokamakMagneticEnv(args.case, randomization="none")
    base = dict(env.case.profile_params)
    rows = []
    grid = [(0.0, 0.0, 0.0)] + [tuple(sign * STEP * (i == j) for j in range(3)) for i in range(3) for sign in (-1, 1)]
    for dlp, dam, dan in grid:
        p = dict(base, paxis=base["paxis"] * float(np.exp(dlp)), alpha_m=base["alpha_m"] + dam,
                 alpha_n=base["alpha_n"] + dan)
        eq = env.reference_eq.create_auxiliary_equilibrium()
        profiles = C.make_profiles(env.case, eq, p)
        try:
            env._solve_static(eq, profiles, dict(env.reference_currents))
            rows.append({"dlog_paxis": dlp, "d_alpha_m": dam, "d_alpha_n": dan, "converged": True,
                         "beta_p": float(eq.poloidalBeta1()), "li3": float(eq.internalInductance3())})
        except EquilibriumError as error:
            rows.append({"dlog_paxis": dlp, "d_alpha_m": dam, "d_alpha_n": dan, "converged": False,
                         "error": str(error)[:120]})
        print(rows[-1], flush=True)
    ok = [r for r in rows if r["converged"]]
    X = np.array([[1.0, r["dlog_paxis"], r["d_alpha_m"], r["d_alpha_n"]] for r in ok])
    coef_bp, *_ = np.linalg.lstsq(X, np.log([r["beta_p"] for r in ok]), rcond=None)
    coef_li, *_ = np.linalg.lstsq(X, np.array([r["li3"] for r in ok]), rcond=None)
    nominal = next(r for r in ok if r["dlog_paxis"] == 0 and r["d_alpha_m"] == 0 and r["d_alpha_n"] == 0)
    result = {
        "case": args.case, "base_profile": base, "nominal": nominal,
        "fit_log_beta_p": dict(zip(["const", "dlog_paxis", "d_alpha_m", "d_alpha_n"], coef_bp.tolist())),
        "fit_li3": dict(zip(["const", "dlog_paxis", "d_alpha_m", "d_alpha_n"], coef_li.tolist())),
        "n_converged": len(ok), "n_total": len(rows), "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
