"""Frozen probes, validation-only selection, explicit final-test switch."""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def metric(y, p, task):
    if task == "awf":
        return {
            "accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
        }
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": r2_score(y, p),
    }


def knn(train, y, query, k=20):
    # Official cosine kNN: softmax-temperature 0.07, k=20, unstandardized embeddings.
    a = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-8)
    b = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
    sim = b @ a.T
    ix = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    vals = np.take_along_axis(sim, ix, axis=1)
    weights = np.exp((vals - vals.max(axis=1, keepdims=True)) / 0.07)
    votes = (np.eye(9)[y[ix].astype(int)] * weights[:, :, None]).sum(axis=1)
    return votes.argmax(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--task", choices=["awf", "lfmc"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--validation-predictions", type=Path)
    a = p.parse_args()
    rows = json.loads(a.labels.read_text())
    z = np.load(a.features, allow_pickle=False)
    if (
        z["ids"].ndim != 1
        or len(z["ids"]) != len(rows)
        or len(z["features"]) != len(rows)
        or len(set(z["ids"].tolist())) != len(rows)
        or set(z["ids"].tolist()) != {r["id"] for r in rows}
    ):
        raise ValueError("IDs missing, duplicate or extra")
    lookup = {s: i for i, s in enumerate(z["ids"].tolist())}
    x = z["features"][[lookup[r["id"]] for r in rows]].astype(np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Nonfinite or invalid features")
    y = np.array([r["label"] for r in rows])
    split = np.array([r["split"] for r in rows])
    tr, va, te = split == "train", split == "val", split == "test"
    if not tr.any() or not va.any():
        raise ValueError("Missing official train or validation split")
    result = {
        "task": a.task,
        "features": str(a.features),
        "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "dimension": x.shape[1],
    }
    constant = (
        float(np.bincount(y[tr].astype(int)).argmax())
        if a.task == "awf"
        else float(y[tr].mean())
    )
    result["constant_val"] = metric(y[va], np.full(va.sum(), constant), a.task)
    candidates = []
    best = None
    grid = (
        [0.001, 0.01, 0.1, 1, 10, 100]
        if a.task == "awf"
        else [0.01, 0.1, 1, 10, 100, 1000, 10000]
    )
    for param in grid:
        estimator = (
            LogisticRegression(C=param, max_iter=3000, random_state=42)
            if a.task == "awf"
            else Ridge(alpha=param)
        )
        model = make_pipeline(StandardScaler(), estimator).fit(x[tr], y[tr])
        scores = metric(y[va], model.predict(x[va]), a.task)
        objective = scores["accuracy"] if a.task == "awf" else -scores["mae"]
        candidates.append({"parameter": param, **scores})
        if best is None or objective > best[0]:
            best = (objective, param, model, scores)
    result["probe"] = {
        "selected_parameter": best[1],
        "validation": best[3],
        "search": candidates,
    }
    if a.validation_predictions:
        a.validation_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            a.validation_predictions,
            ids=np.array([r["id"] for r in rows])[va],
            predictions=best[2].predict(x[va]),
        )
        result["validation_predictions"] = str(a.validation_predictions.resolve())
    if a.task == "awf":
        result["knn_validation"] = metric(y[va], knn(x[tr], y[tr], x[va]), a.task)
    if a.final_test:
        if not te.any():
            raise ValueError(
                "This release has no test split; cannot report a test score"
            )
        result["probe"]["test"] = metric(y[te], best[2].predict(x[te]), a.task)
        result["constant_test"] = metric(y[te], np.full(te.sum(), constant), a.task)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
