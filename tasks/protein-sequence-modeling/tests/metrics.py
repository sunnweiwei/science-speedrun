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
SPEARMAN_TASKS = ("fluorescence", "stability", "beta_lactamase")


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
            "macro_f1": sum(f1s) / n,
            "reward": accuracy}


def score_task(task, truth, found, ppi_affinity_std=None):
    """truth/found: example_id -> label / prediction dict. Raises on malformed input."""
    if set(found) != set(truth):
        raise ValueError("missing or unknown predictions")
    ids = sorted(truth)
    labels = [truth[i] for i in ids]
    predictions = [found[i] for i in ids]
    if task in CLASSES:
        if any(not isinstance(p, dict) or set(p) != {"scores"} for p in predictions):
            raise ValueError("bad classification prediction")
        return classification_metrics(labels, [p["scores"] for p in predictions], CLASSES[task])
    if task in SPEARMAN_TASKS or task == "ppi_affinity":
        if any(not isinstance(p, dict) or set(p) != {"value"} for p in predictions):
            raise ValueError("bad regression prediction")
        values = [finite_number(p["value"]) for p in predictions]
        rho = spearman(labels, values)
        if task == "ppi_affinity":
            rmse = math.sqrt(sum((t - v) ** 2 for t, v in zip(labels, values)) / len(labels))
            return {"rmse": rmse, "spearman": rho,
                    "reward": math.exp(-rmse / max(ppi_affinity_std, 1e-12))}
        return {"spearman": rho, "reward": max(rho, 0.0)}
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
    return {"accuracy": accuracy, "valid_residues": total, "reward": accuracy}


def flatten_metrics(prefix, value, output):
    if isinstance(value, dict):
        for key, item in value.items():
            if key != "invalid_reason":
                flatten_metrics(prefix + "_" + key if prefix else key, item, output)
    elif type(value) in (int, float) and math.isfinite(value):
        output[prefix] = value
