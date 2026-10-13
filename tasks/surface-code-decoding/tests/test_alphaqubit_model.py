"""Small CPU checks of AlphaQubit's mathematical and temporal contracts.

Run with JAX_PLATFORMS=cpu and a bounded CPU affinity. These validate the
independent implementation, not equivalence to unavailable official weights.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baselines.alphaqubit_model import (
    AlphaQubitConfig,
    SyndromeTransformer,
    event_indicators,
    loss_terms,
    perpendicular_pool,
    scatter_grid,
    transformed_model,
)


POSITIONS = np.array([[0, 1], [1, 1], [1, 2], [1, 3],
                      [2, 0], [2, 1], [2, 2], [3, 2]])
GEOMETRY = {"stabilizer_positions": POSITIONS,
            "stabilizer_types": np.sum(POSITIONS, axis=1) % 2,
            "logical_axis": 0, "logical_index": 0}


def small_config(dtype: str = "float32") -> AlphaQubitConfig:
    return AlphaQubitConfig(3, width=4, heads=1, key_size=2, value_size=2,
                           layers=1, conv_width=2, widening=2,
                           feature_layers=1, bias_width=2, bias_layers=1,
                           readout_width=2, readout_layers=1, dtype=dtype)


def sample_features(rounds: int = 3) -> dict[str, jax.Array]:
    random = np.random.default_rng(73)
    values = random.integers(0, 2, size=(2, rounds + 1, 8)).astype(np.float32)
    previous_measurements = np.concatenate((np.zeros_like(values[:, :1]), values[:, :-1]), axis=1)
    events = np.abs(values - previous_measurements)
    previous_events = np.concatenate((np.zeros_like(events[:, :1]), events[:, :-1]), axis=1)
    valid = np.ones((rounds + 1, 8), dtype=bool)
    valid[-1, ::2] = False
    final = np.zeros(rounds + 1, dtype=bool)
    final[-1] = True
    return {key: jnp.asarray(value) for key, value in {
        "measurements": values, "events": events,
        "previous_events": previous_events,
        "valid_mask": valid, "is_final": final,
    }.items()}


def test_spatial_scatter_and_perpendicular_pool_keep_coordinate_convention() -> None:
    values = jnp.arange(2 * 3 * 8 * 2).reshape(2, 3, 8, 2)
    grid = scatter_grid(values, POSITIONS, 3)
    np.testing.assert_array_equal(grid[..., POSITIONS[:, 0], POSITIONS[:, 1], :], values)
    assert grid.shape == (2, 3, 4, 4, 2)
    np.testing.assert_array_equal(grid[..., 0, 0, :], 0)
    # A horizontal logical row has fixed axis 0. Perpendicular pooling averages
    # rows; retaining rows would instead pool along the observable.
    data = jnp.array([[0., 1., 2.], [10., 11., 12.], [20., 21., 22.]])[None, ..., None]
    np.testing.assert_array_equal(perpendicular_pool(data, 0)[0, :, 0], [10., 11., 12.])
    np.testing.assert_array_equal(perpendicular_pool(data, 1)[0, :, 0], [1., 11., 21.])


def test_indicator_features_preserve_directed_time_space_pairs() -> None:
    current = jnp.array([[1., 0., 1.]])
    previous = jnp.array([[0., 1., 1.]])
    result = np.asarray(event_indicators(current, previous))[0]
    np.testing.assert_array_equal(result[0, 1], [0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(result[1, 0], [0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(result[2, 2], np.ones(7))
    np.testing.assert_array_equal(result[0, 0], [1, 0, 0, 0, 1, 0, 0])


def test_attention_matches_supplement_scaled_bias_formula() -> None:
    config = replace(small_config(), width=8, heads=2, key_size=3, value_size=2,
                     conv_width=4, bias_width=4)
    random = np.random.default_rng(912)
    x = random.normal(size=(2, 8, 8)).astype(np.float32)
    fixed = random.normal(size=(8, 8, 4)).astype(np.float32)
    dynamic = random.normal(size=(2, 8, 8, 7)).astype(np.float32)
    model = hk.without_apply_rng(hk.transform(
        lambda value, bias, events: SyndromeTransformer(config, POSITIONS, "block")(value, bias, events)))
    params = hk.data_structures.to_mutable_dict(model.init(jax.random.PRNGKey(12), x, fixed, dynamic))
    # Remove later residual contributions to observe exactly attention + input.
    for module in params:
        if module == "block/dense_down" or module.startswith("block/conv_project_"):
            params[module] = {name: jnp.zeros_like(value) for name, value in params[module].items()}
    # This formula check compares against NumPy FP32 products. GPU defaults
    # may use TF32, whose reduced input precision can exceed these tolerances.
    with jax.default_matmul_precision("highest"):
        actual = np.asarray(model.apply(params, x, fixed, dynamic))
    p = {module: {name: np.asarray(value) for name, value in fields.items()}
         for module, fields in params.items()}
    normalized = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + config.layer_norm_epsilon)
    normalized = normalized * p["block/attention_norm"]["scale"] + p["block/attention_norm"]["offset"]

    def linear(value: np.ndarray, name: str) -> np.ndarray:
        fields = p[f"block/{name}"]
        return value @ fields["w"] + fields.get("b", 0.)

    query = linear(normalized, "query").reshape(2, 8, 2, 3)
    key = linear(normalized, "key").reshape(2, 8, 2, 3)
    value = linear(normalized, "value").reshape(2, 8, 2, 2)
    projected_bias = linear(fixed, "fixed_bias_projection")[None] + linear(dynamic, "event_bias_projection")
    scores = (np.einsum("bihk,bjhk->bhij", query, key) + np.moveaxis(projected_bias, -1, 1)) / np.sqrt(3.)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    expected = x + linear(np.einsum("bhij,bjhv->bihv", weights, value).reshape(2, 8, 4), "attention_output")
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


def test_next_stabilizer_predictions_are_causal_and_terminal_values_are_masked() -> None:
    features = sample_features()
    model = transformed_model(small_config(), GEOMETRY)
    params = model.init(jax.random.PRNGKey(5), features)
    original = model.apply(params, features)
    changed = dict(features)
    for name in ("measurements", "events", "previous_events"):
        changed[name] = features[name].at[:, 2:, :].set(1 - features[name][:, 2:, :])
    future_changed = model.apply(params, changed)
    np.testing.assert_array_equal(original["next_stabilizer_logits"][:, :2],
                                  future_changed["next_stabilizer_logits"][:, :2])
    # Running just the bulk prefix must use the same recurrent parameters and
    # state update, regardless of a later terminal readout branch.
    prefix = {key: value[:2] if key in ("valid_mask", "is_final") else value[:, :2]
              for key, value in features.items()}
    prefix_output = model.apply(params, prefix)
    np.testing.assert_allclose(original["next_stabilizer_logits"][:, :2],
                               prefix_output["next_stabilizer_logits"], rtol=1e-6, atol=1e-6)
    # Off-basis final measurement values are undefined. Their arbitrary storage
    # content must never influence the output through the feature embedding.
    masked = dict(features)
    masked["measurements"] = features["measurements"].at[:, -1, ::2].set(99.)
    masked_output = model.apply(params, masked)
    np.testing.assert_array_equal(original["logits"], masked_output["logits"])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_finite_training_gradients_float32_master_parameters_and_leaf_replay(dtype: str, tmp_path: Path) -> None:
    features = sample_features(rounds=2)
    model = transformed_model(small_config(dtype), GEOMETRY)
    params = model.init(jax.random.PRNGKey(10), features)
    leaves, structure = jax.tree_util.tree_flatten(params)
    assert {value.dtype for value in leaves} == {jnp.dtype("float32")}
    labels = jnp.array([0., 1.])
    objective = lambda candidate: loss_terms(model.apply(candidate, features), features, labels)[0]
    loss, gradients = jax.jit(jax.value_and_grad(objective))(params)
    assert np.isfinite(loss)
    grad_leaves = jax.tree_util.tree_leaves(gradients)
    assert all(np.isfinite(value).all() for value in grad_leaves)
    assert sum(float(jnp.sum(value * value)) for value in grad_leaves) > 0
    # Real float32 master updates survive even when compute activations use
    # bf16. The file below is a trusted test artifact, not the campaign format.
    updated = jax.tree_util.tree_map(lambda weight, grad: weight - 1e-4 * grad, params, gradients)
    updated_leaves = jax.tree_util.tree_leaves(updated)
    assert any(not np.array_equal(before, after) for before, after in zip(leaves, updated_leaves))
    checkpoint = tmp_path / "parameters.npz"
    np.savez(checkpoint, **{str(i): np.asarray(value) for i, value in enumerate(updated_leaves)})
    with np.load(checkpoint, allow_pickle=False) as saved:
        restored = jax.tree_util.tree_unflatten(structure, [jnp.asarray(saved[str(i)]) for i in range(len(leaves))])
    for name, expected in model.apply(updated, features).items():
        actual = model.apply(restored, features)[name]
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, expected)


def test_auxiliary_loss_uses_only_next_bulk_measurements() -> None:
    features = sample_features(rounds=2)
    output = {"logits": jnp.zeros(2), "next_stabilizer_logits": jnp.zeros((2, 3, 8))}
    labels = jnp.array([0., 1.])
    _, metrics = loss_terms(output, features, labels)
    assert float(metrics["aux_loss"]) == pytest.approx(np.log(2.))
    changed = dict(features)
    changed["measurements"] = features["measurements"].at[:, -1].set(99.)
    _, changed_metrics = loss_terms(output, changed, labels)
    assert float(changed_metrics["aux_loss"]) == float(metrics["aux_loss"])
    one_round = sample_features(rounds=1)
    output["next_stabilizer_logits"] = jnp.zeros((2, 2, 8))
    total, metrics = loss_terms(output, one_round, labels)
    assert np.isfinite(total)
    assert float(metrics["aux_loss"]) == 0.
