"""LFMC test reporting only after a locked cohort passes validation confirmation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "environment/evaluation"))
from compare_seeds import summarize_gains  # noqa: E402


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def locked_pairs(lock):
    step = lock["checkpoint_step"]
    return [
        [f"large-random-eval-{lock['policy']['development_seed']}", lock["selected"]]
    ] + [
        [row["paired_random"], f"{row['name']}-step{step}"]
        for row in lock["confirmation"]
    ]


def is_random_initialization(provenance):
    # Early exporter omitted origin; it explicitly recorded checkpoint=null.
    # A missing checkpoint field is not evidence of random initialization.
    return provenance.get("origin") == "random_initialization" or (
        "origin" not in provenance
        and "checkpoint" in provenance
        and provenance["checkpoint"] is None
    )


def check_confirmation(root, lock, comparison):
    if (
        comparison.get("split") != "val"
        or not comparison["tasks"]["lfmc"]["three_seed_criterion_met"]
    ):
        raise ValueError(
            "Validation confirmation has not passed; do not inspect test outcomes"
        )
    result = comparison["tasks"]["lfmc"]
    gains = result["per_seed_gain"]
    if (
        len(gains) < 3
        or not all(gain > 0 for gain in gains)
        or result["spatial_bootstrap_95_percent_interval"][0] <= 0
        or result["seed_and_spatial_bootstrap_95_percent_interval"][0] <= 0
        or abs(float(np.mean(gains)) - result["mean_seed_gain"]) > 1e-10
    ):
        raise ValueError("Validation confirmation flag disagrees with its statistics")
    pairs = locked_pairs(lock)
    context = json.loads((root / "execution-context.json").read_text())
    if (
        comparison["pairs"] != pairs
        or comparison.get("execution_context") != context
        or lock["policy"]["execution_context"] != context
    ):
        raise ValueError("Confirmation cohort or execution context differs from lock")
    expected_seeds = [lock["policy"]["development_seed"]] + [
        r["initialization_seed"] for r in lock["confirmation"]
    ]
    if len(set(expected_seeds)) != len(expected_seeds) or len(expected_seeds) < 3:
        raise ValueError("Need distinct locked initialization seeds")
    if len(gains) != len(expected_seeds):
        raise ValueError("Confirmation statistics have a different seed count")
    actual_loader_seeds = []
    selected_provenance = json.loads(
        (root / "packages" / lock["selected"] / "provenance.json").read_text()
    )
    for seed, (random, trained) in zip(expected_seeds, pairs):
        ps = [
            json.loads((root / "packages" / n / "provenance.json").read_text())
            for n in [random, trained]
        ]
        if (
            not is_random_initialization(ps[0])
            or ps[1]["origin"] != "own_from_scratch_training"
            or any(p.get("training_validity") == "invalid" for p in ps)
        ):
            raise ValueError("Ineligible paired artifacts")
        if (
            any(
                ps[0][k] != ps[1][k]
                for k in [
                    "variant",
                    "parameters",
                    "initialization_seed",
                    "infer_sha256",
                ]
            )
            or ps[1]["initialization_seed"] != seed
        ):
            raise ValueError("Unpaired model initialization or architecture")
        if ps[1]["training_recipe"] != lock["training_recipe"]:
            raise ValueError("Training recipe differs from confirmation lock")
        if any(
            ps[1][key] != selected_provenance[key]
            for key in ["variant", "parameters", "infer_sha256"]
        ):
            raise ValueError("Architecture differs across locked seeds")
        actual_loader_seeds.append(ps[1]["data_seed"])
        if (
            trained == lock["selected"]
            and ps[1]["encoder_sha256"] != lock["encoder_sha256"]
        ):
            raise ValueError("Selected checkpoint hash differs from lock")
        for name, provenance in zip([random, trained], ps):
            if (
                digest(root / "packages" / name / "encoder.pt")
                != provenance["encoder_sha256"]
            ):
                raise ValueError("Encoder weights changed")
    expected_loader_seeds = [selected_provenance["data_seed"]] + [
        row["data_seed"] for row in lock["confirmation"]
    ]
    if (
        actual_loader_seeds != expected_loader_seeds
        or len(set(actual_loader_seeds)) != len(actual_loader_seeds)
        or comparison.get("initialization_seeds") != expected_seeds
        or comparison.get("data_seeds") != actual_loader_seeds
    ):
        raise ValueError("Loader or initialization seeds differ from confirmation")
    return pairs, expected_seeds, context


def predict(root, name, rows, y, train, validation, test, *, raw=False):
    report_path = root / "results" / f"{name}-lfmc.json"
    report = json.loads(report_path.read_text())
    if "test" in report["probe"]:
        raise ValueError("Require original validation-only probe selection")
    path = Path(report["features"])
    if (
        not raw
        and path.resolve()
        != (root / "features" / f"{name}-lfmc" / "features.npz").resolve()
    ):
        raise ValueError("Cannot mix feature execution contexts")
    values = np.load(path, allow_pickle=False)
    ids = values["ids"].tolist()
    if (
        len(ids) != len(rows)
        or len(set(ids)) != len(rows)
        or set(ids) != {r["id"] for r in rows}
    ):
        raise ValueError("Feature IDs missing, duplicate or extra")
    lookup = {v: i for i, v in enumerate(ids)}
    x = values["features"][[lookup[r["id"]] for r in rows]].astype(np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid features")
    alpha = report["probe"]["selected_parameter"]
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[train], y[train])
    val_mae = float(np.abs(model.predict(x[validation]) - y[validation]).mean())
    if abs(val_mae - report["probe"]["validation"]["mae"]) > 1e-7:
        raise ValueError("Refit disagrees with locked validation probe")
    predicted = model.predict(x[test])
    errors = np.abs(predicted - y[test])
    return (
        errors,
        predicted,
        {
            "selected_alpha_from_validation": alpha,
            "validation_mae_check": val_mae,
            "test_mae": float(errors.mean()),
            "feature_sha256": digest(path),
            "validation_report_sha256": digest(report_path),
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--validation-comparison", type=Path, required=True)
    p.add_argument("--raw-baseline", default="large-raw")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    predictions_dir = a.output.parent / (a.output.stem + "-predictions")
    if a.output.exists() or predictions_dir.exists():
        p.error("Preserve previous test outcomes; use a fresh output")
    lock = json.loads(a.lock.read_text())
    comparison = json.loads(a.validation_comparison.read_text())
    pairs, seeds, context = check_confirmation(a.data_root, lock, comparison)
    labels_path = a.data_root / "eval/lfmc/labels.json"
    rows = json.loads(labels_path.read_text())
    y = np.array([r["label"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    train, validation, test = splits == "train", splits == "val", splits == "test"
    if not all(m.any() for m in [train, validation, test]):
        raise ValueError("Missing released split")
    sites = np.round(np.array([r["latlon"] for r in rows])[test], 6)
    unique, inverse = np.unique(sites, axis=0, return_inverse=True)
    errors, predictions, records = {}, {}, {}
    for name in [n for pair in pairs for n in pair] + [a.raw_baseline]:
        errors[name], predictions[name], records[name] = predict(
            a.data_root,
            name,
            rows,
            y,
            train,
            validation,
            test,
            raw=name == a.raw_baseline,
        )
        if name != a.raw_baseline:
            records[name]["encoder_sha256"] = json.loads(
                (a.data_root / "packages" / name / "provenance.json").read_text()
            )["encoder_sha256"]
    summary = summarize_gains(
        [errors[r] - errors[t] for r, t in pairs], inverse, len(unique), "lfmc", test
    )
    # Validation remains the acceptance criterion; test is a separate locked-cohort report.
    summary["test_three_seed_positive_interval"] = summary.pop(
        "three_seed_criterion_met"
    )
    raw = summarize_gains(
        [errors[a.raw_baseline] - errors[t] for _, t in pairs],
        inverse,
        len(unique),
        "lfmc",
        test,
    )
    raw["test_three_seed_positive_interval"] = raw.pop("three_seed_criterion_met")
    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 42,
        "pairs": pairs,
        "initialization_seeds": seeds,
        "execution_context": context,
        "models": records,
        "paired_random_comparison": summary,
        "raw_baseline_comparison": raw,
        "lock_sha256": digest(a.lock),
        "validation_comparison_sha256": digest(a.validation_comparison),
        "labels_sha256": digest(labels_path),
        "analysis_sha256": digest(Path(__file__)),
        "bootstrap_implementation_sha256": digest(
            Path(__file__).resolve().parents[2] / "environment/evaluation/compare_seeds.py"
        ),
        "limitation": "No checkpoint or probe parameter selected using these test outcomes. LFMC test was previously exposed by Tiny; this is not a fresh blind holdout.",
    }
    predictions_dir.mkdir(parents=True)
    test_ids = np.array([r["id"] for r in rows])[test]
    for name, values in predictions.items():
        np.savez_compressed(
            predictions_dir / f"{name}.npz", ids=test_ids, predictions=values
        )
    report["prediction_directory"] = str(predictions_dir)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
