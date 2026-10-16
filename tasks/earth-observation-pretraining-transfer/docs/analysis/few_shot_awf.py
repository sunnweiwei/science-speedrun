"""Secondary label-efficiency diagnostic; never count label seeds as pretrain seeds."""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = json.loads((a.data_root / "eval/awf/labels.json").read_text())
    ids = np.array([r["id"] for r in rows])
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    train = np.array([r["split"] == "train" for r in rows])
    val = np.array([r["split"] == "val" for r in rows])
    cohorts = {}
    for count in [5, 20]:
        for seed in [42, 43, 44]:
            rng = np.random.default_rng(seed)
            selected = np.concatenate(
                [
                    rng.permutation(np.flatnonzero(train & (y == label)))[:count]
                    for label in np.unique(y[train])
                ]
            )
            assert train[selected].all()
            cohorts[f"k{count}-labels{seed}"] = selected
    report = {
        "protocol": "secondary AWF label-efficiency diagnostic",
        "validation_rows": int(val.sum()),
        "pretraining_initialization_seeds": 1,
        "stable_pretraining_claim": False,
        "limitation": "Label-selection seeds are not pretraining seeds; adaptive validation development, no independent holdout.",
        "cohorts": {
            key: {"rows": len(ix), "train_ids": ids[ix].tolist()}
            for key, ix in cohorts.items()
        },
        "validation_ids": ids[val].tolist(),
        "models": {},
    }
    for name in a.names:
        source = json.loads((a.data_root / "results" / f"{name}-awf.json").read_text())
        z = np.load(source["features"], allow_pickle=False)
        lookup = {key: i for i, key in enumerate(z["ids"].tolist())}
        if len(lookup) != len(ids) or set(lookup) != set(ids):
            raise ValueError("Feature IDs do not match the complete AWF release")
        x = z["features"][[lookup[key] for key in ids]].astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite features")
        results = {}
        for key, selected in cohorts.items():
            best, search = None, []
            for c in [0.001, 0.01, 0.1, 1, 10, 100]:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, max_iter=3000, random_state=42),
                )
                model.fit(x[selected], y[selected])
                prediction = model.predict(x[val])
                metric = {
                    "C": c,
                    "accuracy": float(accuracy_score(y[val], prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y[val], prediction)
                    ),
                }
                search.append(metric)
                if best is None or metric["accuracy"] > best["accuracy"]:
                    best = dict(metric, predictions=prediction.tolist())
            results[key] = dict(best, search=search)
        report["models"][name] = results
        print(
            name, {key: value["accuracy"] for key, value in results.items()}, flush=True
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
