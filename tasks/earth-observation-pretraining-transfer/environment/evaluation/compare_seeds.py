"""Paired validation comparisons, conditional on the declared training recipe."""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prediction(root, name, task, rows, train, validation, y, enforce_context=True):
    result = json.loads((root / "results" / f"{name}-{task}.json").read_text())
    if enforce_context and (root / "execution-context.json").exists():
        expected = {
            "features": root / "features" / f"{name}-{task}" / "features.npz",
            "validation_predictions": root
            / "results"
            / f"{name}-{task}-predictions.npz",
        }
        if any(
            key not in result or Path(result[key]).resolve() != path.resolve()
            for key, path in expected.items()
        ):
            raise ValueError(
                "Cannot mix execution contexts in paired encoder comparisons"
            )
    if "test" in result["probe"]:
        raise ValueError("Use validation-only reports")
    if "validation_predictions" in result:
        data = np.load(result["validation_predictions"], allow_pickle=False)
        ids = [row["id"] for row in rows if row["split"] == "val"]
        if data["ids"].tolist() != ids:
            raise ValueError("Prediction IDs are not aligned")
        predicted = data["predictions"]
        errors = (
            np.abs(predicted - y[validation])
            if task == "lfmc"
            else (predicted != y[validation]).astype(float)
        )
        expected = (
            result["probe"]["validation"]["mae"]
            if task == "lfmc"
            else 1 - result["probe"]["validation"]["accuracy"]
        )
        if abs(errors.mean() - expected) > 1e-8:
            raise ValueError("Saved predictions disagree with score")
        return errors
    if task == "awf":
        raise ValueError(
            "AWF requires saved predictions; refitting can change near-boundary decisions"
        )
    data = np.load(result["features"], allow_pickle=False)
    ids = data["ids"].tolist()
    if len(set(ids)) != len(rows) or set(ids) != {r["id"] for r in rows}:
        raise ValueError("Feature IDs do not match evaluation release")
    lookup = {key: index for index, key in enumerate(ids)}
    x = data["features"][[lookup[row["id"]] for row in rows]].astype(np.float64)
    parameter = result["probe"]["selected_parameter"]
    estimator = (
        Ridge(alpha=parameter)
        if task == "lfmc"
        else LogisticRegression(C=parameter, max_iter=3000, random_state=42)
    )
    model = make_pipeline(StandardScaler(), estimator).fit(x[train], y[train])
    predicted = model.predict(x[validation])
    return (
        np.abs(predicted - y[validation])
        if task == "lfmc"
        else (predicted != y[validation]).astype(float)
    )


def summarize_gains(gains, inverse, group_count, task, validation):
    gains = np.stack(gains)
    rng = np.random.default_rng(42)
    samples = rng.integers(0, group_count, size=(5000, group_count))
    totals = np.bincount(inverse, weights=gains.mean(axis=0))
    counts = np.bincount(inverse)
    bootstrap = totals[samples].sum(axis=1) / counts[samples].sum(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    seed_totals = np.stack([np.bincount(inverse, weights=gain) for gain in gains])
    seed_draws = rng.integers(0, len(gains), size=(5000, len(gains)))
    joint = seed_totals[seed_draws[:, :, None], samples[:, None, :]].sum(
        axis=(1, 2)
    ) / (len(gains) * counts[samples].sum(axis=1))
    joint_interval = np.quantile(joint, [0.025, 0.975]).tolist()
    return {
        "metric_reduction": "mae" if task == "lfmc" else "error_rate",
        "groups": group_count,
        "rows": int(validation.sum()),
        "per_seed_gain": gains.mean(axis=1).tolist(),
        "mean_seed_gain": float(gains.mean()),
        "spatial_bootstrap_95_percent_interval": interval,
        "seed_and_spatial_bootstrap_95_percent_interval": joint_interval,
        "three_seed_criterion_met": len(gains) >= 3
        and bool((gains.mean(axis=1) > 0).all())
        and interval[0] > 0
        and joint_interval[0] > 0,
    }


def validate_confirmation_recipes(recipes):
    """Keep single-pair diagnostics available; require complete declared locks."""
    if len(recipes) <= 1:
        return
    if any(
        isinstance(recipe, dict) and recipe.get("recipe_complete") is False
        for recipe in recipes
    ):
        raise ValueError("Confirmation requires complete training recipe provenance")
    if any(recipe is None for recipe in recipes) or any(
        recipe != recipes[0] for recipe in recipes
    ):
        raise ValueError("Confirmation runs must have the same locked training recipe")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="random-name:trained-name for each paired seed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-baseline", help="Optional raw-feature report name; reported separately"
    )
    args = parser.parse_args()
    pairs = [pair.split(":") for pair in args.pairs]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("Expected random:trained pairs")
    seeds = [
        json.loads(
            (args.data_root / "packages" / trained / "provenance.json").read_text()
        )["initialization_seed"]
        for _, trained in pairs
    ]
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "Repeated initialization seeds cannot count as independent runs"
        )
    recipes = [
        json.loads(
            (args.data_root / "packages" / trained / "provenance.json").read_text()
        ).get("training_recipe")
        for _, trained in pairs
    ]
    validate_confirmation_recipes(recipes)
    loader_seeds = [
        json.loads(
            (args.data_root / "packages" / trained / "provenance.json").read_text()
        ).get("data_seed")
        for _, trained in pairs
    ]
    if len(pairs) > 1 and (
        any(seed is None for seed in loader_seeds)
        or len(set(loader_seeds)) != len(loader_seeds)
    ):
        raise ValueError("Confirmation runs require distinct recorded loader seeds")
    report = {
        "pairs": pairs,
        "initialization_seeds": seeds,
        "data_seeds": loader_seeds,
        "split": "val",
        "bootstrap_replicates": 5000,
        "limitation": "Conditional on adaptive validation development; not independent holdout proof.",
        "tasks": {},
    }
    context = args.data_root / "execution-context.json"
    if context.exists():
        report["execution_context"] = json.loads(context.read_text())
    for task in ("awf", "lfmc"):
        rows = json.loads((args.data_root / "eval" / task / "labels.json").read_text())
        y = np.array([row["label"] for row in rows])
        train = np.array([row["split"] == "train" for row in rows])
        validation = np.array([row["split"] == "val" for row in rows])
        coords = np.array([row["latlon"] for row in rows])[validation]
        # LFMC: resample sites; AWF: 0.1-degree spatial blocks, not individual nearby pixels.
        groups = np.round(coords, 6) if task == "lfmc" else np.floor(coords / 0.1)
        unique, inverse = np.unique(groups, axis=0, return_inverse=True)
        gains = []
        trained_errors = []
        for random, trained in pairs:
            provenance = [
                json.loads(
                    (args.data_root / "packages" / name / "provenance.json").read_text()
                )
                for name in (random, trained)
            ]
            if any(p.get("training_validity") == "invalid" for p in provenance):
                raise ValueError(
                    "Known invalid training implementation cannot count as a candidate"
                )
            if (
                provenance[0]["checkpoint"] is not None
                or provenance[1]["checkpoint"] is None
            ):
                raise ValueError(
                    "Expected random initialization and own trained checkpoint"
                )
            if provenance[1].get("origin") == "official_public_pretrained_reference":
                raise ValueError(
                    "Public pretrained references cannot count as our from-scratch runs"
                )
            for key in ("variant", "parameters", "initialization_seed", "infer_sha256"):
                if provenance[0][key] != provenance[1][key]:
                    raise ValueError(f"Unpaired {key}")
            trained_errors.append(
                prediction(args.data_root, trained, task, rows, train, validation, y)
            )
            gains.append(
                prediction(args.data_root, random, task, rows, train, validation, y)
                - trained_errors[-1]
            )
        report["tasks"][task] = summarize_gains(
            gains, inverse, len(unique), task, validation
        )
        if args.raw_baseline:
            raw_errors = prediction(
                args.data_root,
                args.raw_baseline,
                task,
                rows,
                train,
                validation,
                y,
                enforce_context=False,
            )
            report["tasks"][task]["raw_baseline_comparison"] = {
                "baseline": args.raw_baseline,
                **summarize_gains(
                    [raw_errors - errors for errors in trained_errors],
                    inverse,
                    len(unique),
                    task,
                    validation,
                ),
            }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
