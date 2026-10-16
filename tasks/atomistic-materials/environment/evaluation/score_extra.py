"""Score a run of extra.py against the references of the same split.

    python score_extra.py --suite kappa|mof|amorphous|cleavage|neb --run RUN_DIR \
        --inputs INPUTS_DIR --references REFERENCES_DIR

INPUTS_DIR is the label-free split directory given to extra.py; REFERENCES_DIR holds its
references.json(.gz). The two may be the same directory (development builds).

Missing or failed items are scored, never dropped:
- kappa: SRME as in Matbench Discovery (imaginary modes, broken symmetry and failures score 2).
- mof: MAPE of B0; each MOF's error is capped at 200 %, and a failed MOF scores the cap.
- amorphous: per-system energy RMSE (offset removed) and force RMSE, each relative to the reference
  standard deviation; a failed structure scores about twice the error of predicting the mean
  (energy error +-2 std, forces -F_ref).
- cleavage: relative MAE (MAE / mean |reference|); each slab's error is capped at 2 |reference|, and
  a failed slab scores the cap.
- neb: barrier MAE over all paths with each error capped at 2 eV (diverging NEB runs give errors of
  tens of eV); a failed path scores the cap. Also reports the converged fraction and the MAE over
  converged paths.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

KAPPA_ERROR_MAX = 2.0
MOF_CAP = 2.0
NEB_CAP_EV = 2.0


def load(path: Path):
    return json.load(gzip.open(path)) if path.suffix == ".gz" else json.loads(path.read_text())


def predictions(run: Path) -> dict[str, dict]:
    out = {}
    for f in (run / "items").glob("*.json"):
        r = json.loads(f.read_text())
        out[r["id"]] = r
    return out


def ok(p):
    return p is not None and p.get("status") == "ok"


# ------------------------------------------------------------------------------------------ kappa
def kappa_avg(kappa) -> np.ndarray:
    a = np.asarray(kappa, dtype=float)
    if a.shape[-2:] == (3, 3):
        return np.trace(a, axis1=-2, axis2=-1) / 3
    if a.shape[-1] == 6:
        return a[..., :3].mean(axis=-1)
    if a.shape[-1] == 3:
        return a.mean(axis=-1)
    raise ValueError(f"unexpected kappa shape {a.shape}")


def _finite(value, nonnegative=False):
    try:
        v = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if v.size == 0 or not np.all(np.isfinite(v)) or (nonnegative and np.any(v < 0)):
        return None
    return v


def srme_one(pred: dict | None, ref: dict) -> tuple[float, float, bool]:
    """(SRME, SRD, censored) for one material, following matbench_discovery/metrics/phonons.py."""
    true_total = float(np.ravel(ref["kappa_tot_avg"])[0])
    true_mode = np.asarray(ref["mode_kappa_tot_avg"], dtype=float)
    if not ok(pred):
        return KAPPA_ERROR_MAX, -KAPPA_ERROR_MAX, True
    censored = bool(pred.get("has_imag_ph_modes"))
    expected_spg = ref.get("init_spg_num")
    if not censored and pred.get("final_spg_num") and expected_spg:
        censored = pred["final_spg_num"] != expected_spg
    total = _finite(kappa_avg(pred["kappa_tot_rta"]) if pred.get("kappa_tot_rta") is not None else None, True)
    srd = -KAPPA_ERROR_MAX
    if total is not None and total.size == 1:
        t = float(total.ravel()[0])
        srd = 0.0 if t + true_total == 0 else KAPPA_ERROR_MAX * (t - true_total) / (t + true_total)
    if censored or total is None or total.size != 1:
        return KAPPA_ERROR_MAX, srd, True
    mode = _finite(kappa_avg(pred["mode_kappa_tot_rta"]) if pred.get("mode_kappa_tot_rta") is not None else None)
    # Predictions are produced in the process that imports the untrusted calculator.
    # The quadrature normalizer belongs to the pinned reference, never that process.
    weights = _finite(ref.get("weights"), True)
    if mode is None or weights is None or weights.sum() <= 0 or mode.shape != true_mode.shape:
        return KAPPA_ERROR_MAX, srd, True
    micro = np.abs(mode - true_mode).sum(axis=tuple(range(1, mode.ndim))) / weights.sum()
    denom = np.atleast_1d(total) + np.atleast_1d(true_total)
    score = float(np.ravel(np.where(denom == 0, np.where(micro == 0, 0, KAPPA_ERROR_MAX),
                                    KAPPA_ERROR_MAX * micro / denom))[0])
    if not np.isfinite(score) or not 0 <= score <= KAPPA_ERROR_MAX:
        return KAPPA_ERROR_MAX, srd, True
    return score, srd, False


def score_kappa(pred, root):
    ref = {r["material_id"]: r for r in load(root / "references.json.gz")}
    rows = {m: srme_one(pred.get(m), r) for m, r in sorted(ref.items())}
    srme = np.array([r[0] for r in rows.values()]); srd = np.array([r[1] for r in rows.values()])
    return {"items": len(rows), "kappa_SRME": float(srme.mean()), "kappa_SRE": float(np.abs(srd).mean()),
            "kappa_SRD": float(srd.mean()), "censored_or_failed": int(sum(r[2] for r in rows.values()))}


# ------------------------------------------------------------------------------------------ others
def score_mof(pred, root):
    ref = {r["id"]: float(r["B0_GPa"]) for r in load(root / "references.json")}
    err, failed = [], 0
    for k, y in sorted(ref.items()):
        p = pred.get(k)
        b = p.get("B_GPa") if ok(p) else None
        if b is None or not np.isfinite(b):
            failed += 1; err.append(MOF_CAP)
        else:
            err.append(min(abs(b - y) / abs(y), MOF_CAP))
    return {"items": len(ref), "bulk_modulus_MAPE_pct": float(100 * np.mean(err)), "failed": failed}


def score_amorphous(pred, root):
    refs = load(root / "references.json.gz")
    systems = sorted({r["system"] for r in refs})
    out, failed = {"items": len(refs)}, 0
    for s in systems:
        rs = [r for r in refs if r["system"] == s]
        er = np.array([r["energy_per_atom"] for r in rs])
        d, fr, fp = [], [], []
        for k, r in enumerate(rs):
            p = pred.get(r["id"])
            f_ref = np.asarray(r["forces"], dtype=float)
            if ok(p) and np.isfinite(p["energy_per_atom"]) and np.shape(p["forces"]) == f_ref.shape \
                    and np.isfinite(p["forces"]).all():
                d.append(p["energy_per_atom"] - r["energy_per_atom"]); fp.append(np.asarray(p["forces"], dtype=float))
            else:  # about twice the trivial predictor's error, so failing never helps
                failed += 1; d.append((1 if k % 2 == 0 else -1) * 2 * er.std()); fp.append(-f_ref)
            fr.append(f_ref)
        d = np.array(d)
        d = d - d.mean()
        fr, fp = np.concatenate(fr), np.concatenate(fp)
        out[f"{s}_energy_RelRMSE_pct"] = float(100 * np.sqrt((d ** 2).mean()) / er.std())
        out[f"{s}_force_RelRMSE_pct"] = float(100 * np.sqrt(((fp - fr) ** 2).mean()) / fr.std())
    out["energy_RelRMSE_pct_mean"] = float(np.mean([out[f"{s}_energy_RelRMSE_pct"] for s in systems]))
    out["force_RelRMSE_pct_mean"] = float(np.mean([out[f"{s}_force_RelRMSE_pct"] for s in systems]))
    out["failed"] = failed
    return out


def score_cleavage(pred, root):
    ref = {r["id"]: float(r["cleavage_energy"]) for r in load(root / "references.json")}
    y = np.array(list(ref.values()))
    good = np.array([ok(pred.get(k)) and np.isfinite(pred[k]["cleavage_energy"]) for k in ref])
    p = np.array([pred[k]["cleavage_energy"] if g else np.nan for k, g in zip(ref, good)])
    err = np.where(good, np.minimum(np.abs(p - y), 2 * np.abs(y)), 2 * np.abs(y))
    return {"items": len(ref), "cleavage_relMAE_pct": float(100 * err.mean() / np.abs(y).mean()),
            "cleavage_MAE_eV_A2": float(err.mean()), "cleavage_signed_eV_A2": float((p - y)[good].mean()) if good.any() else None,
            "failed": int((~good).sum())}


def score_neb(pred, root):
    ref = {r["id"]: float(r["barrier_eV"]) for r in load(root / "references.json")}
    y = np.array(list(ref.values()))
    good = np.array([ok(pred.get(k)) and np.isfinite(pred[k]["barrier_eV"]) for k in ref])
    p = np.array([pred[k]["barrier_eV"] if g else np.nan for k, g in zip(ref, good)])
    err = np.where(good, np.minimum(np.abs(p - y), NEB_CAP_EV), NEB_CAP_EV)
    conv = np.array([g and bool(pred[k]["converged"]) for k, g in zip(ref, good)])
    return {"items": len(ref), "barrier_MAE_eV": float(err.mean()),
            "barrier_MAE_converged_eV": float(np.abs(p - y)[conv].mean()) if conv.any() else None,
            "converged_frac": float(conv.mean()), "failed": int(len(ref) - sum(good))}


SCORERS = {"kappa": score_kappa, "mof": score_mof, "amorphous": score_amorphous,
           "cleavage": score_cleavage, "neb": score_neb}


def score(suite: str, run: Path, inputs: Path, references: Path) -> dict:
    info = json.loads((run / "run.json").read_text())
    import hashlib
    if info.get("inputs_manifest_sha256") != hashlib.sha256((inputs / "manifest.json").read_bytes()).hexdigest():
        raise SystemExit(f"{suite}: run was made on a different split")
    return {"suite": suite, **SCORERS[suite](predictions(run), references), "complete": info["complete"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", choices=sorted(SCORERS), required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--references", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(score(a.suite, a.run, a.inputs, a.references), indent=1))


if __name__ == "__main__":
    main()
