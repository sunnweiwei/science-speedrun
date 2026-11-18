"""Trusted PEER task scoring; pure Python, never imports a submission.

Kept byte-identical with environment/evaluation/metrics.py so development
self-evaluation matches the grader.
"""
import math

TASKS = ("fluorescence", "stability", "beta_lactamase", "solubility",
         "subcellular_localization", "binary_localization", "fold",
         "secondary_structure", "yeast_ppi", "human_ppi", "ppi_affinity")
CLASSES = {"solubility": 2, "subcellular_localization": 10, "binary_localization": 2,
           "fold": 1195, "yeast_ppi": 2, "human_ppi": 2}
# Chance-calibration class denominators matching harness/ssi_index_v3.py:SEQ_K.
# `fold` prediction vectors span all 1,195 SCOPe fold labels (`CLASSES["fold"] = 1195`), while
# the held-out `test_fold_holdout` split contains 136 observed classes with positive test support.
# Because `balanced_accuracy` averages per-class recall strictly over classes present in `truth`
# (`tp + fn > 0`), any constant or uniform predictor achieves `1 / 136` on the held-out test split.
SEQ_K = {"solubility": 2, "subcellular_localization": 10, "binary_localization": 2,
         "fold": 136, "yeast_ppi": 2, "human_ppi": 2}
SEQ_GROUPS = {
    "Function": ("fluorescence", "stability", "beta_lactamase", "solubility"),
    "Localization": ("subcellular_localization", "binary_localization"),
    "Structure": ("fold", "secondary_structure"),
    "Protein interactions": ("yeast_ppi", "human_ppi", "ppi_affinity"),
}
SPEARMAN_TASKS = ("fluorescence", "stability", "beta_lactamase")
PPI_SYMMETRIC_TASKS = ("yeast_ppi", "human_ppi")
SS3_MAJORITY_BASELINE = 0.4306464618
PPI_AFFINITY_ANCHOR_RMSE = 3.083206134
SYMMETRIC_TIE_TOL = 1e-12
NON_FLATTENED_KEYS = {"invalid_reason", "pair_probabilities"}


def probabilities(values, n):
    if (not isinstance(values, list) or len(values) != n
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values)
            or sum(values) <= 0):
        raise ValueError("invalid scores")
    total = sum(values)
    return [x / total for x in values]


def finite_number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid number")
    return float(value)


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        rank = (start + stop) / 2 + 1
        for index in range(start, stop + 1):
            ranks[order[index]] = rank
        start = stop + 1
    return ranks


def spearman(truth, predicted):
    if len(truth) < 2:
        return 0.0
    a, b = average_ranks(truth), average_ranks(predicted)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    scale = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return covariance / scale if scale > 0 else 0.0


def classification_metrics(truth, raw_scores, n):
    guesses = [max(range(n), key=probabilities(scores, n).__getitem__) for scores in raw_scores]
    recalls, f1s = [], []
    for label in range(n):
        tp = sum(t == label and g == label for t, g in zip(truth, guesses))
        fp = sum(t != label and g == label for t, g in zip(truth, guesses))
        fn = sum(t == label and g != label for t, g in zip(truth, guesses))
        if tp + fn:
            recalls.append(tp / (tp + fn))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    accuracy = sum(t == g for t, g in zip(truth, guesses)) / len(truth)
    return {"accuracy": accuracy,
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
            "macro_f1": sum(f1s) / n}


def task_skill(task, task_metrics):
    """Exact SSI V3 signed task skill (`harness/ssi_index_v3.py:sequence`)."""
    if task in SEQ_K:
        x = finite_number(task_metrics["balanced_accuracy"])
        k = SEQ_K[task]
        return (x - 1.0 / k) / (1.0 - 1.0 / k)
    if task == "secondary_structure":
        x = finite_number(task_metrics["accuracy"])
        return (x - SS3_MAJORITY_BASELINE) / (1.0 - SS3_MAJORITY_BASELINE)
    if task == "ppi_affinity":
        x = finite_number(task_metrics["rmse"])
        return 1.0 - x / PPI_AFFINITY_ANCHOR_RMSE
    if task in SPEARMAN_TASKS:
        return finite_number(task_metrics["spearman"])
    raise ValueError(f"unknown task: {task}")


def score_task(task, truth, found, ppi_affinity_std=None, found_swapped=None):
    """truth/found: example_id -> label / prediction dict. Raises on malformed input."""
    if set(found) != set(truth):
        raise ValueError("missing or unknown predictions")
    ids = sorted(truth)
    labels = [truth[i] for i in ids]
    predictions = [found[i] for i in ids]
    if task in CLASSES:
        if any(not isinstance(p, dict) or set(p) != {"scores"} for p in predictions):
            raise ValueError("bad classification prediction")
        if task in PPI_SYMMETRIC_TASKS and found_swapped is not None:
            if set(found_swapped) != set(truth):
                raise ValueError("missing or unknown swapped PPI predictions")
            predictions_swapped = [found_swapped[i] for i in ids]
            if any(not isinstance(p, dict) or set(p) != {"scores"} for p in predictions_swapped):
                raise ValueError("bad swapped classification prediction")
            ab_probs = [probabilities(p["scores"], 2) for p in predictions]
            ba_probs = [probabilities(p["scores"], 2) for p in predictions_swapped]
            sym_scores = []
            pair_probs = {}
            p_ab_vals, p_ba_vals, p_sym_vals, asym_diffs = [], [], [], []
            for eid, pab, pba in zip(ids, ab_probs, ba_probs):
                q_sym = 0.5 * (pab[0] + pba[0])
                p_sym = 0.5 * (pab[1] + pba[1])
                if abs(p_sym - q_sym) <= SYMMETRIC_TIE_TOL:
                    sym_vec = [0.5, 0.5]
                    p_sym_clean = 0.5
                else:
                    sym_vec = [q_sym, p_sym]
                    p_sym_clean = p_sym
                sym_scores.append(sym_vec)
                p_ab_vals.append(pab[1])
                p_ba_vals.append(pba[1])
                p_sym_vals.append(p_sym_clean)
                asym_diffs.append(abs(pab[1] - pba[1]))
                pair_probs[eid] = {"p_ab": pab[1], "p_ba": pba[1], "p_sym": p_sym_clean}
            res = classification_metrics(labels, sym_scores, 2)
            ab_res = classification_metrics(labels, [p["scores"] for p in predictions], 2)
            ba_res = classification_metrics(labels, [p["scores"] for p in predictions_swapped], 2)
            n_pairs = max(len(ids), 1)
            res.update({
                "p_ab_mean": sum(p_ab_vals) / n_pairs,
                "p_ba_mean": sum(p_ba_vals) / n_pairs,
                "p_sym_mean": sum(p_sym_vals) / n_pairs,
                "asymmetry_mae": sum(asym_diffs) / n_pairs,
                "ab_accuracy": ab_res["accuracy"],
                "ab_balanced_accuracy": ab_res["balanced_accuracy"],
                "ba_accuracy": ba_res["accuracy"],
                "ba_balanced_accuracy": ba_res["balanced_accuracy"],
                "symmetric_ppi": True,
                "pair_probabilities": pair_probs,
            })
        else:
            res = classification_metrics(labels, [p["scores"] for p in predictions], CLASSES[task])
            if task in PPI_SYMMETRIC_TASKS:
                res["symmetric_ppi"] = False
        skill = task_skill(task, res)
        res["skill"] = skill
        res["reward"] = skill
        return res
    if task in SPEARMAN_TASKS or task == "ppi_affinity":
        if any(not isinstance(p, dict) or set(p) != {"value"} for p in predictions):
            raise ValueError("bad regression prediction")
        values = [finite_number(p["value"]) for p in predictions]
        rho = spearman(labels, values)
        if task == "ppi_affinity":
            rmse = math.sqrt(sum((t - v) ** 2 for t, v in zip(labels, values)) / len(labels))
            res = {"rmse": rmse, "spearman": rho}
        else:
            res = {"spearman": rho}
        skill = task_skill(task, res)
        res["skill"] = skill
        res["reward"] = skill
        return res
    # secondary_structure
    correct = total = 0
    for label, prediction in zip(labels, predictions):
        if not isinstance(prediction, dict) or set(prediction) != {"ss3_scores"}:
            raise ValueError("bad ss3 prediction")
        scores = prediction["ss3_scores"]
        if not isinstance(scores, list) or len(scores) != len(label["ss3"]):
            raise ValueError("ss3 prediction length mismatch")
        for target, valid, residue_scores in zip(label["ss3"], label["valid_mask"], scores):
            guess = max(range(3), key=probabilities(residue_scores, 3).__getitem__)
            if valid:
                total += 1
                correct += guess == target
    if not total:
        raise ValueError("no valid residues")
    accuracy = correct / total
    res = {"accuracy": accuracy, "valid_residues": total}
    skill = task_skill(task, res)
    res["skill"] = skill
    res["reward"] = skill
    return res


def aggregate_pillars(details):
    """Aggregate 11 PEER tasks into the 4 equal-weight SSI V3 capability pillars."""
    skills = {}
    for names in SEQ_GROUPS.values():
        for task in names:
            entry = details.get(task, {})
            if entry.get("invalid_reason") is not None or "skill" not in entry:
                skills[task] = 0.0
            else:
                skills[task] = finite_number(entry["skill"])
    pillars = {
        pillar: sum(skills[t] for t in names) / len(names)
        for pillar, names in SEQ_GROUPS.items()
    }
    signed_score = sum(pillars.values()) / len(pillars)
    score = max(0.0, min(1.0, signed_score))
    return {
        "score": score,
        "reward": score,
        "signed_score": signed_score,
        "pillars": pillars,
        "task_skills": skills,
    }


def flatten_metrics(prefix, value, output):
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in NON_FLATTENED_KEYS:
                flatten_metrics(prefix + "_" + key if prefix else key, item, output)
    elif type(value) in (int, float) and math.isfinite(value):
        output[prefix] = value
