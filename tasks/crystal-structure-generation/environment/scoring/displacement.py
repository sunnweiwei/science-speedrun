"""Structure-level relaxation displacement, reported separately from the raw-geometry reward."""
from hashlib import sha256
import math
import random
from statistics import mean

MODELS = ("orb", "mace", "uma")
SAMPLE_SIZE = 250


def sample_slots(requests):
    slots = [f"{key}--{index:06d}" for key, request in requests.items()
             for index in range(request["n_samples"])]
    def priority(slot):
        return sha256(("crystalgen-displacement/v1/" + slot).encode()).digest()
    return sorted(sorted(slots, key=priority)[:SAMPLE_SIZE])


def summarize(slots, measured):
    if not set(measured).issubset(slots):
        raise RuntimeError("Relaxation diagnostic returned an unexpected sample ID")
    values, per_sample = [], {}
    for slot in slots:
        raw = measured.get(slot, {})
        models = {k: v for k in MODELS if isinstance(v := raw.get(k), (int, float)) and math.isfinite(v) and v >= 0}
        value = mean(models.values()) if len(models) == len(MODELS) else None
        if value is not None:
            values.append(value)
        per_sample[slot] = {"mlip_rmsd_A": models, "ensemble_rmsd_A": value}
    ci = None
    if len(values) > 1:
        rng = random.Random(20261115)
        boot = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
        ci = [boot[49], boot[1949]]
    return {"definition": "Equal mean of ORB-v3, MACE-MP and UMA per-structure Cartesian relaxation RMSDs (angstroms, LeMat Eq. 7).",
            "protocol": "crystalgen-displacement/v1", "fmax_eV_per_A": 0.02, "max_steps": 50,
            "selected_slots": len(slots), "valid_selected": len(measured), "complete_triplets": len(values),
            "mean_A": mean(values) if values else None, "bootstrap_ci95_A": ci,
            "ci_scope": "Resample selected structures with three finite displacements; missing slots and MLIP failures are counted separately.",
            "samples": per_sample}
