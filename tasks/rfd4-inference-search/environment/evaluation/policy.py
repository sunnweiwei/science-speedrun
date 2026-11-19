"""Submission contract for the three target-free monomer problems."""
import json
import math
from pathlib import Path

LENGTHS = {100: "short", 150: "medium", 250: "long"}
PASSES_PER_SLOT = 180
SECONDS_PER_SLOT = 36
CLASH = "clash_metrics.frac_atoms_clashing_1p5A"


class InvalidPolicy(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidPolicy(message)


def integer(n, low, high):
    return type(n) is int and low <= n <= high


def unique_keys(pairs):
    obj = dict(pairs)
    require(len(obj) == len(pairs), "Duplicate JSON key")
    return obj


def check_priority(expr, depth=0):
    require(depth <= 5, "Priority expression is too deep")
    if isinstance(expr, str):
        require(expr in ("quality", "rank", "parent", "draw"), "Unknown priority feature")
    elif type(expr) in (int, float):
        require(-16 <= expr <= 16 and math.isfinite(expr), "Invalid priority constant")
    else:
        require(isinstance(expr, list) and len(expr) == 3 and expr[0] in ("+", "-", "*", "min", "max"),
                "Expected a binary priority expression")
        check_priority(expr[1], depth + 1)
        check_priority(expr[2], depth + 1)


def cost(plan):
    """Return native trunk passes per delivered slot, charging all siblings and lookaheads."""
    fields = {"bands", "width", "checkpoints"}
    require(isinstance(plan, dict) and set(plan) in (fields, fields | {"solver"}), "Invalid plan fields")
    require(plan.get("solver", "stochastic-tsr") in ("stochastic-tsr", "ode-tsr"), "Invalid TSR solver")
    bands = plan["bands"]
    require(isinstance(bands, list) and 1 <= len(bands) <= 60, "Invalid recycle schedule")
    depths = []
    for pair in bands:
        require(isinstance(pair, list) and len(pair) == 2, "Each band is [steps, trunk passes per step]")
        n, k = pair
        require(integer(n, 1, 180) and integer(k, 1, 8), "Invalid recycle band")
        depths.extend([k] * n)
    steps = len(depths)
    require(20 <= steps <= 180, "Schedule must contain 20 to 180 steps")
    width = plan["width"]
    require(integer(width, 1, 5) and width in (1, 2, 5), "Width must divide the development and final slot counts")
    checkpoints = plan["checkpoints"]
    require(isinstance(checkpoints, list) and len(checkpoints) <= 8, "At most eight checkpoints")
    require(checkpoints or width == 1, "Independent sampling uses width one")
    spent = last = 0
    for point in checkpoints:
        require(isinstance(point, dict) and set(point) in ({"step", "branches", "score"}, {"step", "branches", "score", "priority"}), "Invalid checkpoint fields")
        if "priority" in point:
            check_priority(point["priority"])
        t, b, mode = point["step"], point["branches"], point["score"]
        require(integer(t, last + 1, steps) and integer(b, 1, 8), "Invalid checkpoint step or branches")
        require(mode in ("clean", "rollout") and not (mode == "clean" and t == steps), "Invalid scoring mode")
        spent += b * sum(depths[last:t])
        if mode == "rollout":
            spent += b * sum(depths[t:])
        last = t
    spent += sum(depths[last:])
    require(spent <= PASSES_PER_SLOT, f"Estimated {spent} trunk passes per slot exceeds {PASSES_PER_SLOT}")
    return spent


def load_policy(package):
    f = Path(package) / "policy.json"
    require(not f.is_symlink() and f.is_file() and f.stat().st_size <= 65536, "Expected regular policy.json, at most 64 KiB")
    try:
        doc = json.loads(f.read_bytes(), object_pairs_hook=unique_keys)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InvalidPolicy("Invalid JSON") from exc
    require(isinstance(doc, dict) and set(doc) == {"version", "default", "by_length"}, "Invalid top-level fields")
    require(integer(doc["version"], 1, 1), "Unsupported policy version")
    overrides = doc["by_length"]
    require(isinstance(overrides, dict) and set(overrides) <= set(map(str, LENGTHS)), "Unknown length")
    cost(doc["default"])
    plans = {n: overrides.get(str(n), doc["default"]) for n in LENGTHS}
    for plan in plans.values():
        cost(plan)
    return plans


def native_config(plan, length, slots, seed, assets, run):
    cost(plan)
    width = plan["width"]
    require(slots > 0 and slots % width == 0, "Slot count must be a positive multiple of beam width")
    steps = sum(n for n, _ in plan["bands"])
    widest = max((p["branches"] for p in plan["checkpoints"]), default=1) * width
    batch = min(5, slots // width, max(1, 10 // widest))
    cfg = {
        "defaults": ["inference", "_self_"],
        "inputs": str(assets / "monomers" / f"monomer-{LENGTHS[length]}.cif.zst"),
        "out_dir": str(run),
        "hf_ckpt_path": str(assets / "rfd4"),
        "compile_lengths": None, "num_workers": 0, "seed": seed,
        "batch_size": batch, "num_replicates": slots // width,
        "skip_existing": False,
        "generation": {"args": {"nsteps": steps}, "n_recycle_schedule": plan["bands"]},
        "metrics_tags": [],
    }
    if plan.get("solver") == "ode-tsr":
        cfg["generation"]["model"] = {mode: {"gt": {"mode": "ode"}} for mode in ("bb_ca", "local_latents")}
    if plan["checkpoints"]:
        cfg["search"] = {
            "beam_width": width, "n_branch": 1, "save_rollouts": False,
            "checkpoints": {str(p["step"]): {
                "metrics": ["clash_metrics"],
                "rewards": {CLASH: {"weight": -1.0}},
                "n_branch": p["branches"],
                "use_clean_prediction": p["score"] == "clean",
            } for p in plan["checkpoints"]}
        }
    return cfg
