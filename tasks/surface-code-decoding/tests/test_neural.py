"""CPU numerical checks for native model fidelity and checkpoint portability.

Run with CUDA_VISIBLE_DEVICES=-1 to keep these small checks off training GPUs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from baselines.neural import (
    EXPECTED_PARAMETER_COUNTS,
    REFERENCE_DIR,
    UPSTREAM_COMMIT,
    build_model,
    model_config,
)


def test_vendored_sources_match_pinned_hashes() -> None:
    provenance = json.loads((REFERENCE_DIR / "provenance.json").read_text())
    assert provenance["commit"] == UPSTREAM_COMMIT
    assert provenance["modifications"] == []
    for name, digest in provenance["files_sha256"].items():
        assert hashlib.sha256((REFERENCE_DIR / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("distance", [3, 5])
def test_native_model_training_and_checkpoint(distance: int, tmp_path: Path) -> None:
    tf.keras.backend.clear_session()
    with tf.device("/CPU:0"):
        model = build_model(distance, seed=74)
        assert model.count_params() == EXPECTED_PARAMETER_COUNTS[distance]
        assert model.output_names == ["main_output", "aux_output"]
        assert model.input_shape == [
            (None, None, distance**2 - 1),
            (None, (distance**2 - 1) // 2),
        ]
        random = np.random.default_rng(99)
        inputs = {
            "rec_input": random.integers(
                0, 2, size=(8, 3, distance**2 - 1)
            ).astype(np.float32),
            "eval_input": random.integers(
                0, 2, size=(8, (distance**2 - 1) // 2)
            ).astype(np.float32),
        }
        labels = np.array([[0], [1]] * 4, dtype=np.float32)
        targets = {"main_output": labels, "aux_output": labels}
        initial_weight = model.trainable_variables[0].numpy().copy()
        losses = model.train_on_batch(inputs, targets, return_dict=True)
        assert all(np.isfinite(value) for value in losses.values())
        assert not np.array_equal(initial_weight, model.trainable_variables[0].numpy())
        expected = model(inputs, training=False)
        assert all(np.isfinite(output.numpy()).all() for output in expected)

        checkpoint = tmp_path / f"d{distance}.weights.h5"
        model.save_weights(checkpoint)
        restored = build_model(distance, seed=18)
        restored.load_weights(checkpoint)
        actual = restored(inputs, training=False)
        for before, after in zip(expected, actual):
            np.testing.assert_allclose(before.numpy(), after.numpy(), rtol=0, atol=1e-7)


def test_width_override_and_invalid_architectures() -> None:
    assert model_config(5, width=128)["model"]["LSTM"]["units"] == [128, 128]
    with pytest.raises(ValueError, match="distances"):
        model_config(7)
    for width in (0, -1, True, 64.5):
        with pytest.raises(ValueError, match="width"):
            model_config(3, width=width)
    for learning_rate in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="learning_rate"):
            model_config(3, learning_rate=learning_rate)
