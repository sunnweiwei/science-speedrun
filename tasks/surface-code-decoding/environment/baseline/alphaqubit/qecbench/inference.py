"""Generic frozen-weight AlphaQubit inference for arbitrary sample IDs/order/N."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
from qecbench.data import condition_from_record, contained, load_condition, load_manifest, write_json


def predict(submission: Path, inputs: Path, output: Path, batch_size: int = 256,
            group: str | None = None) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifest = load_manifest(inputs)
    config = json.loads((submission / "model_config.json").read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    from baselines.alphaqubit_artifact import GEOMETRY_KEYS, load_model
    from baselines.alphaqubit_data import AlphaQubitLayout
    import jax
    import jax.numpy as jnp
    selected = [row for row in manifest["conditions"] if group is None or row["group"] == group]
    if not selected:
        raise ValueError("No selected conditions")
    completed = []
    for group_name in dict.fromkeys(row["group"] for row in selected):
        record = config["models"][group_name]
        model, parameters, metadata = load_model(contained(submission, record["weights"]), contained(submission, record["metadata"]))
        parameters = jax.device_put(parameters)
        apply = jax.jit(lambda params, features: jax.nn.sigmoid(model.apply(params, features)["logits"].astype(jnp.float32)))
        for row in (item for item in selected if item["group"] == group_name):
            arrays = load_condition(inputs, row)
            layout = AlphaQubitLayout.from_condition(condition_from_record(inputs, row))
            actual = layout.metadata()
            if any(actual[key] != metadata["geometry"][key] for key in GEOMETRY_KEYS):
                raise ValueError("Model/input geometry mismatch")
            values = np.empty(row["shots"], dtype=np.float32)
            for begin in range(0, len(values), batch_size):
                end = min(begin + batch_size, len(values))
                packed = arrays["detectors"][begin:end]
                if len(packed) < batch_size:
                    packed = np.pad(packed, ((0, batch_size - len(packed)), (0, 0)))
                predicted = np.asarray(apply(parameters, layout.decode_packed_features(packed)), dtype=np.float32)
                if predicted.shape != (batch_size,) or not np.isfinite(predicted).all() or np.any((predicted < 0) | (predicted > 1)):
                    raise ValueError("Model returned invalid probabilities")
                values[begin:end] = predicted[:end - begin]
            np.savez(output / f"{row['name']}.npz", sample_ids=arrays["sample_ids"], probabilities=values)
            completed.append(row["name"])
        del apply, parameters, model
        jax.clear_caches()
    result = {"schema": "surface-code-decoding-predictions/v1", "conditions": completed}
    write_json(output / "predictions.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--group", help="Optional subset for parallel reference replay")
    args = parser.parse_args()
    print(json.dumps(predict(args.submission, args.input, args.output, args.batch_size, args.group)))


if __name__ == "__main__":
    main()
