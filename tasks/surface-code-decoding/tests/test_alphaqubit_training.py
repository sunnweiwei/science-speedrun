"""Tests for sampling/optimizer resume, LER selection and anchored LAMB.

Run with JAX_PLATFORMS=cpu to include the bitwise resume contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import stim

from baselines.alphaqubit_model import AlphaQubitConfig
from baselines.alphaqubit_train import (
    FACTORS, add_anchored_decay, argument_parser, curriculum_probabilities,
    fit_validation, learning_rate, load_internal, load_portable,
    make_update, publish_best, run_training,
)


def small_config() -> AlphaQubitConfig:
    return AlphaQubitConfig(3, width=4, heads=1, key_size=2, value_size=2,
                           layers=1, conv_width=2, widening=2, feature_layers=1,
                           bias_width=2, bias_layers=1, readout_width=2, readout_layers=1)


def make_data(root: Path) -> None:
    root.mkdir()
    for rounds in range(1, 26, 2):
        directory = root / f"surface_code_bX_d3_r{rounds:02d}_center_3_5"
        directory.mkdir()
        ideal = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=rounds)
        noisy = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=rounds,
                                       after_clifford_depolarization=0.01, before_measure_flip_probability=0.02)
        ideal.to_file(directory / "circuit_ideal.stim")
        noisy.detector_error_model(decompose_errors=True).to_file(directory / "pij_from_even_for_odd.dem")


def args_for(data: Path, output: Path, steps: int, resume: bool = False):
    return argument_parser().parse_args([
        "--data-root", str(data), "--group", "d3_bX_center_3_5", "--output", str(output),
        "--hours", "1", "--max-steps", str(steps), "--microbatch", "2", "--eval-shots", "4",
        "--eval-seconds", "99999", "--seed", "17", "--allow-cpu", *( ["--resume"] if resume else []),
    ])


def assert_tree_equal(left, right) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_paper_schedules_and_literal_batch_counter_default() -> None:
    probabilities = curriculum_probabilities(12_800_000)
    assert probabilities.argmax() == 0  # Logistic midpoint f=0.5.
    assert probabilities[0] / probabilities[-1] == pytest.approx(13.0)
    assert curriculum_probabilities(200_000_000).argmax() == len(FACTORS) - 1
    assert np.sum(probabilities) == pytest.approx(1.0)
    assert learning_rate(3, 0) == pytest.approx(3.46e-8)
    assert learning_rate(5, 9999) == pytest.approx(2.45e-4)
    assert learning_rate(5, 80000) == pytest.approx(2.45e-4 * 0.7)
    assert learning_rate(5, 2000000) == pytest.approx(2.45e-4 * 0.7**5)
    args = argument_parser().parse_args(["--data-root", "/tmp/data", "--group", "g", "--output", "/tmp/out", "--hours", "1"])
    assert args.batch_change_counter == "steps" and args.microbatch == 256


def test_anchored_nonbias_decay_and_microbatch_accumulation() -> None:
    params = {"linear": {"w": jnp.array([0.25, -0.5]), "b": jnp.array([0.125]), "offset": jnp.array([0.2])}}
    anchor = jax.tree.map(lambda value: value * 0.5, params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    direction = add_anchored_decay(zeros, params, anchor, 0.08)
    np.testing.assert_allclose(direction["linear"]["w"], [0.01, -0.02])
    np.testing.assert_array_equal(direction["linear"]["b"], [0.])
    np.testing.assert_array_equal(direction["linear"]["offset"], [0.])
    assert_tree_equal(add_anchored_decay(zeros, params, params, 0.08), zeros)

    class LinearModel:
        @staticmethod
        def apply(weights, features):
            logits = features["measurements"][:, 0, :] @ weights["linear"]["w"] + weights["linear"]["b"][0]
            return {"logits": logits, "next_stabilizer_logits": jnp.zeros((len(logits), 2, 2))}

    measurements = jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2) / 8
    features = {"measurements": measurements, "events": measurements, "previous_events": measurements,
                "valid_mask": jnp.ones((2, 2), dtype=bool), "is_final": jnp.array([False, True])}
    labels = jnp.array([[0.], [1.], [0.], [1.]])
    outcomes = []
    # Compare the mathematical accumulation at FP32 precision: differently
    # shaped GPU matmuls may otherwise select different TF32 kernels.
    with jax.default_matmul_precision("highest"):
        for microbatch in (1, 4):
            adam, update = make_update(LinearModel(), microbatch, 0.08)
            outcomes.append(update(params, adam.init(params), params, anchor, features, labels, jnp.array(1e-3)))
    for a, b in zip(jax.tree.leaves(outcomes[0]), jax.tree.leaves(outcomes[1]), strict=True):
        np.testing.assert_allclose(a, b, rtol=3e-6, atol=2e-7)
    updated, _, ema, _ = outcomes[0]
    expected_ema = jax.tree.map(lambda old, new: old + 0.0001 * (new - old), params, updated)
    assert_tree_equal(ema, expected_ema)


def counts_for_fit(intercept: float = 0.002) -> list[dict]:
    rows = []
    for rounds in range(1, 26, 2):
        fidelity = np.exp(intercept + rounds * np.log(1 - 2 * 0.02))
        total = 1_000_000
        rows.append({"condition": f"surface_code_bX_d3_r{rounds:02d}_center_3_5",
                     "rounds": rounds, "errors": int(round((1 - fidelity) / 2 * total)), "shots": total})
    return rows


def test_validation_gate_never_selects_chance_or_bad_intercept() -> None:
    fit = fit_validation(counts_for_fit())
    assert fit["accepted"] and fit["logical_error_per_round"] == pytest.approx(0.02, abs=1e-6)
    assert fit["selection_r_squared"] > 0.999
    low_intercept = fit_validation(counts_for_fit(-0.2))
    assert not low_intercept["accepted"]
    assert "intercept_below_paper_cutoff" in low_intercept["rejection_reasons"]
    chance = counts_for_fit()
    chance[-1]["errors"] = chance[-1]["shots"] // 2
    invalid = fit_validation(chance)
    assert not invalid["accepted"] and invalid["logical_error_per_round"] is None


@pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="Bitwise resume requires JAX_PLATFORMS=cpu; GPU JIT reductions are not bitwise stable.",
)
def test_native_training_resume_reproduces_optimizer_rng_data_and_portable_params(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_data(data)
    interrupted, continuous = tmp_path / "resumed", tmp_path / "continuous"
    options = dict(config_override=small_config(), initial_batch_size=4, final_batch_size=8)
    first = run_training(args_for(data, interrupted, 2), **options)
    assert first["global_steps"] == 2 and first["seen_samples"] == 8 and first["stop_reason"] == "max_steps"
    before = load_internal(interrupted / "last.pkl")
    assert int(before["opt_state"].count) == 2
    with pytest.raises(FileExistsError, match="Nonempty"):
        run_training(args_for(data, interrupted, 3), **options)
    resumed = run_training(args_for(data, interrupted, 4, resume=True), **options)
    full = run_training(args_for(data, continuous, 4), **options)
    a, b = load_internal(interrupted / "last.pkl"), load_internal(continuous / "last.pkl")
    for key in ("params", "ema", "opt_state", "jax_rng"):
        assert_tree_equal(a[key], b[key])
    assert a["numpy_rng"] == b["numpy_rng"] and a["samplers"] == b["samplers"]
    assert resumed["global_steps"] == full["global_steps"] == 4
    assert resumed["seen_samples"] == full["seen_samples"] == 16
    assert int(a["opt_state"].count) == 4
    assert resumed["elapsed_seconds"] > first["elapsed_seconds"]
    metadata = json.loads((interrupted / "metadata.json").read_text())
    assert len(metadata["sessions"]) == 2 and all(session.get("ended_at") for session in metadata["sessions"])
    assert metadata["sessions"][1]["resume_seen_samples"] == 8
    # Portable export is tested with an explicit synthetic accepted-selection
    # fixture; tiny random models are not misrepresented as valid LER fits.
    a["best_params"] = a["params"]
    a["best_validation"] = {"global_steps": 4, "seen_samples": 16, "fit": fit_validation(counts_for_fit())}
    publish_best(interrupted, a, metadata["identity"])
    restored, description = load_portable(interrupted / "best.npz")
    assert_tree_equal(restored, a["params"])
    assert description["selected_LER"] == pytest.approx(0.02, abs=1e-6)
    # Changing a semantic identity or the source fingerprints cannot resume.
    altered = args_for(data, interrupted, 5, resume=True)
    altered.seed += 1
    with pytest.raises(ValueError, match="identity changed"):
        run_training(altered, **options)
    changed_initializer = args_for(data, interrupted, 5, resume=True)
    changed_initializer.init_model = interrupted / "best.npz"
    with pytest.raises(ValueError, match="initializer checkpoint SHA changed"):
        run_training(changed_initializer, **options)
