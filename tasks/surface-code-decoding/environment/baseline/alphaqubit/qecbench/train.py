"""Minimal fine-tuning example for one specialist of a packaged submission (for instance one
trained from scratch with baselines.alphaqubit_train and packaged with qecbench.package).

This intentionally simple AdamW/BCE recipe is a development example, not the
historical LAMB/EMA recipe or a claim of improved accuracy. It modifies only the
selected specialist and preserves the other nine frozen starting models.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from typing import Any
import numpy as np
from qecbench.data import DevelopmentDataset, sha256, write_json


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("steps, batch size and learning rate must be positive")
    if args.output.exists():
        raise ValueError("Output must be a new directory")
    import jax
    import jax.numpy as jnp
    import optax
    from baselines.alphaqubit_artifact import load_model
    config = json.loads((args.starter / "model_config.json").read_text())
    record = config["models"][args.group]
    model, parameters, metadata = load_model(args.starter / record["weights"], args.starter / record["metadata"])
    training = DevelopmentDataset(args.data, "train", args.group)
    validation = DevelopmentDataset(args.data, "validation", args.group)
    optimizer = optax.adamw(args.learning_rate, weight_decay=0.0)
    state = optimizer.init(parameters)

    @jax.jit
    def update(params: Any, opt_state: Any, features: Any, labels: Any) -> tuple[Any, Any, Any]:
        def loss(p: Any) -> Any:
            logits = model.apply(p, features)["logits"].astype(jnp.float32)
            return optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        value, gradients = jax.value_and_grad(loss)(params)
        changes, next_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, changes), next_state, value

    rng = np.random.default_rng(args.seed)
    losses = []
    for step in range(args.steps):
        features, labels = training.batch(rng, args.batch_size)
        parameters, state, loss = update(parameters, state, features, labels)
        losses.append(float(loss))
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}), flush=True)
    infer = jax.jit(lambda params, features: jax.nn.sigmoid(model.apply(params, features)["logits"].astype(jnp.float32)))
    errors, totals = [], []
    for index in range(len(validation.records)):
        layout, arrays = validation.get(index)
        limit = min(args.validation_shots, len(arrays["labels"]))
        count = 0
        for begin in range(0, limit, args.batch_size):
            end = min(begin + args.batch_size, limit)
            packed = arrays["detectors"][begin:end]
            if len(packed) < args.batch_size:
                packed = np.pad(packed, ((0, args.batch_size - len(packed)), (0, 0)))
            probabilities = np.asarray(infer(parameters, layout.decode_packed_features(packed)))[:end - begin]
            count += int(np.count_nonzero((probabilities >= 0.5) != arrays["labels"][begin:end]))
        errors.append(count)
        totals.append(limit)
    shutil.copytree(args.starter, args.output, ignore=shutil.ignore_patterns("__pycache__", ".venv"))
    destination = args.output / record["weights"]
    np.savez(destination, **{f"{module}/{name}": np.asarray(value, dtype=np.float32)
                            for module, leaves in parameters.items() for name, value in leaves.items()})
    metadata["checkpoint_sha256"] = sha256(destination)
    metadata["release_warm_start"] = {"steps": args.steps, "optimizer": "AdamW", "learning_rate": args.learning_rate,
                                       "batch_size": args.batch_size, "seed": args.seed,
                                       "source_checkpoint_sha256": sha256(args.starter / record["weights"])}
    write_json(args.output / record["metadata"], metadata)
    model_manifest = json.loads((args.output / "model_manifest.json").read_text())
    for checkpoint in model_manifest["checkpoints"]:
        checkpoint["sha256"] = sha256(args.output / checkpoint["path"])
    write_json(args.output / "model_manifest.json", model_manifest)
    result = {"group": args.group, "steps": args.steps, "last_training_loss": losses[-1],
              "validation_mean_condition_failure": float(np.mean(np.array(errors) / totals)),
              "validation_shots_per_condition": args.validation_shots,
              "data_manifest_sha256": sha256(args.data / "train/manifest.json"),
              "validation_manifest_sha256": sha256(args.data / "validation/manifest.json"),
              "starting_checkpoint_sha256": sha256(args.starter / record["weights"]),
              "recipe": "AdamW BCE example; no test data or test selection"}
    write_json(args.output / "training_provenance.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--starter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--validation-shots", type=int, default=5120)
    args = parser.parse_args()
    if args.validation_shots <= 0:
        parser.error("validation shots must be positive")
    print(json.dumps(train(args)))


if __name__ == "__main__":
    main()
