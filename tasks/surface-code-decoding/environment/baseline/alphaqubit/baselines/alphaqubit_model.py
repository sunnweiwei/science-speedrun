"""Independent JAX/Haiku implementation of AlphaQubit (Nature 2024).

Algorithms 1–5 in the supplement specify the recurrent core. Methods and
Extended Data Figs. 4/8 specify the embedder, attention bias, readout and widths.
Unspecified details are explicit in ``implementation_choices``; this module is
not an official Google implementation and does not imply paper-level accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np


@dataclass(frozen=True)
class AlphaQubitConfig:
    distance: int
    width: int = 320
    layers: int = 3
    heads: int = 4
    key_size: int = 32
    value_size: int = 32
    conv_width: int = 160
    widening: int = 5
    feature_layers: int = 2
    bias_width: int = 48
    bias_layers: int = 8
    attention_bias: bool = True
    readout_width: int = 64
    readout_layers: int = 16
    readout_widening: int = 1
    dtype: str = "float32"
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.distance not in (3, 5):
            raise ValueError("This campaign supports audited Sycamore d3/d5 only")
        if self.width % self.heads or (self.width * self.widening) % 2:
            raise ValueError("Invalid head or gated-dense dimensions")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("Unsupported compute dtype")

    @property
    def dilations(self) -> tuple[int, int, int]:
        return (1, 1, 1) if self.distance == 3 else (1, 1, 2)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "dilations": list(self.dilations),
                "implementation_choices": {
                    "value_head_size": self.value_size,
                    "value_size_basis": "Haiku MultiHeadAttention default equals key_size; consistent with paper scaling parameter count",
                    "generic_resnet": "pre-LayerNorm, Linear-GELU-Linear, additive residual",
                    "attention_scaling": "(QK^T + bias) / sqrt(key_size), Supplement Algorithm1",
                    "cycle_embedding": "linear projection of log1p(bulk_rounds); encoding unspecified in Fig4f",
                    "diagonal_indicators": "diagonal matrices for current^2, current*previous, previous^2",
                    "padding": "zero spatial padding; optional learned padding omitted",
                    "initialization": "Haiku Linear/Conv defaults; learned embeddings truncated normal std1",
                    "layer_norm": "learned scale/offset; epsilon explicitly configured",
                    "gelu": "exact erf GELU",
                }}


def norm(x: jax.Array, name: str, epsilon: float) -> jax.Array:
    # Accumulating moments in float32 also permits a stable bf16 compute mode.
    y = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                     eps=epsilon, name=name)(x.astype(jnp.float32))
    return y.astype(x.dtype)


class ResidualMLP(hk.Module):
    def __init__(self, width: int, layers: int, widening: int = 1,
                 epsilon: float = 1e-5, name: str | None = None):
        super().__init__(name=name)
        self.width, self.layers, self.widening, self.epsilon = width, layers, widening, epsilon

    def __call__(self, x: jax.Array) -> jax.Array:
        for i in range(self.layers):
            y = norm(x, f"norm_{i}", self.epsilon)
            y = jax.nn.gelu(hk.Linear(self.width * self.widening, name=f"up_{i}")(y), approximate=False)
            x = x + hk.Linear(self.width, name=f"down_{i}")(y)
        return x


def scatter_grid(x: jax.Array, positions: np.ndarray, distance: int) -> jax.Array:
    grid = jnp.zeros((*x.shape[:-2], distance + 1, distance + 1, x.shape[-1]), dtype=x.dtype)
    return grid.at[..., positions[:, 0], positions[:, 1], :].set(x)


def perpendicular_pool(data: jax.Array, logical_axis: int) -> jax.Array:
    """Pool across the fixed coordinate of the logical path in [B,row,col,C]."""
    return jnp.mean(data, axis=logical_axis + 1)


def event_indicators(current: jax.Array, previous: jax.Array) -> jax.Array:
    outer = lambda a, b: a[..., :, None] * b[..., None, :]
    cc, cp, pc, pp = outer(current, current), outer(current, previous), outer(previous, current), outer(previous, previous)
    eye = jnp.eye(current.shape[-1], dtype=current.dtype)
    return jnp.stack((cc, cp, pc, pp, cc * eye, cp * eye, pp * eye), axis=-1)


class AttentionBias(hk.Module):
    def __init__(self, config: AlphaQubitConfig, positions: np.ndarray,
                 stabilizer_types: np.ndarray, name: str = "attention_bias"):
        super().__init__(name=name)
        self.config, self.positions, self.types = config, positions, stabilizer_types

    def __call__(self) -> jax.Array:
        c, pos = self.config, self.positions
        size = c.distance + 1
        s = len(pos)
        xi, yi = np.broadcast_to(pos[:, 0, None], (s, s)), np.broadcast_to(pos[:, 1, None], (s, s))
        xj, yj = xi.T, yi.T
        entries = [("xi", xi, size), ("yi", yi, size), ("xj", xj, size), ("yj", yj, size),
                   ("dx", xi - xj + c.distance, 2 * c.distance + 1),
                   ("dy", yi - yj + c.distance, 2 * c.distance + 1),
                   ("manhattan", np.abs(xi - xj) + np.abs(yi - yj), 2 * c.distance + 1),
                   ("same_basis", (self.types[:, None] == self.types[None, :]).astype(np.int32), 2)]
        x = sum(hk.Embed(vocab, c.bias_width, name=name)(jnp.asarray(ids)) for name, ids, vocab in entries)
        return ResidualMLP(c.bias_width, c.bias_layers, epsilon=c.layer_norm_epsilon, name="resnet")(x)


class SyndromeTransformer(hk.Module):
    def __init__(self, config: AlphaQubitConfig, positions: np.ndarray, name: str):
        super().__init__(name=name)
        self.c, self.positions = config, positions

    def __call__(self, x: jax.Array, static_bias: jax.Array | None,
                 indicators: jax.Array | None) -> jax.Array:
        c = self.c
        b, s, _ = x.shape
        y = norm(x, "attention_norm", c.layer_norm_epsilon)
        q = hk.Linear(c.heads * c.key_size, name="query")(y).reshape(b, s, c.heads, c.key_size)
        k = hk.Linear(c.heads * c.key_size, name="key")(y).reshape(b, s, c.heads, c.key_size)
        v = hk.Linear(c.heads * c.value_size, name="value")(y).reshape(b, s, c.heads, c.value_size)
        scores = jnp.einsum("bihk,bjhk->bhij", q, k).astype(jnp.float32)
        if c.attention_bias:
            # Concatenation followed by one bias-free linear projection is
            # algebraically equivalent to these two independent projections.
            fixed = hk.Linear(c.heads, with_bias=False, name="fixed_bias_projection")(static_bias)
            dynamic = hk.Linear(c.heads, with_bias=False, name="event_bias_projection")(indicators)
            scores = scores + jnp.moveaxis(fixed[None] + dynamic, -1, 1).astype(jnp.float32)
        attention = jax.nn.softmax(scores / math.sqrt(c.key_size), axis=-1).astype(x.dtype)
        y = jnp.einsum("bhij,bjhv->bihv", attention, v).reshape(b, s, c.heads * c.value_size)
        x = x + hk.Linear(c.width, name="attention_output")(y)
        y = norm(x, "dense_norm", c.layer_norm_epsilon)
        a, gate = jnp.split(hk.Linear(c.width * c.widening, name="dense_up")(y), 2, axis=-1)
        y = jax.nn.gelu(a, approximate=False) * gate
        x = x + hk.Linear(c.width, name="dense_down")(y)
        grid = scatter_grid(x, self.positions, c.distance)
        for i, dilation in enumerate(c.dilations):
            y = norm(grid, f"conv_norm_{i}", c.layer_norm_epsilon)
            y = hk.Conv2D(c.conv_width, 3, rate=dilation, padding="SAME", name=f"conv_{i}")(y)
            y = jax.nn.gelu(y, approximate=False)
            if c.conv_width != c.width:
                y = hk.Conv2D(c.width, 1, name=f"conv_project_{i}")(y)
            grid = grid + y
        return grid[:, self.positions[:, 0], self.positions[:, 1], :]


class AlphaQubit(hk.Module):
    def __init__(self, config: AlphaQubitConfig, positions: np.ndarray,
                 stabilizer_types: np.ndarray, logical_axis: int, logical_index: int,
                 name: str = "alphaqubit"):
        super().__init__(name=name)
        self.c, self.positions, self.types = config, np.asarray(positions), np.asarray(stabilizer_types)
        self.logical_axis, self.logical_index = int(logical_axis), int(logical_index)
        expected = config.distance ** 2 - 1
        if self.positions.shape != (expected, 2) or self.types.shape != (expected,):
            raise ValueError("Geometry does not match the code distance")
        if len(np.unique(self.positions, axis=0)) != expected or np.any(self.positions < 0) or np.any(self.positions > config.distance):
            raise ValueError("Invalid stabilizer grid positions")
        if logical_axis not in (0, 1) or not 0 <= logical_index < config.distance:
            raise ValueError("Invalid logical observable location")

    def __call__(self, features: dict[str, jax.Array]) -> dict[str, jax.Array]:
        c = self.c
        dtype = jnp.bfloat16 if c.dtype == "bfloat16" else jnp.float32
        measurements = features["measurements"].astype(dtype)
        events = features["events"].astype(dtype)
        previous = features["previous_events"].astype(dtype)
        b, t, s = events.shape
        final = features["is_final"][None, :, None, None]
        valid = features["valid_mask"][None, :, :, None]
        bulk = (hk.Linear(c.width, name="measurement_projection")(measurements[..., None])
                + hk.Linear(c.width, name="event_projection")(events[..., None]))
        final_values = (hk.Linear(c.width, name="final_measurement_projection")(measurements[..., None])
                        + hk.Linear(c.width, name="final_event_projection")(events[..., None]))
        missing = hk.get_parameter("missing_final", (c.width,), dtype=jnp.float32, init=hk.initializers.TruncatedNormal(1.0)).astype(dtype)
        embedded = jnp.where(final, jnp.where(valid, final_values, missing), bulk)
        embedded = embedded + hk.Embed(s, c.width, name="stabilizer_index")(jnp.arange(s))[None, None].astype(dtype)
        embedded = ResidualMLP(c.width, c.feature_layers, epsilon=c.layer_norm_epsilon, name="feature_resnet")(embedded)
        bias = AttentionBias(c, self.positions, self.types)() if c.attention_bias else None
        layers = [SyndromeTransformer(c, self.positions, name=f"syndrome_{i}") for i in range(c.layers)]
        next_stabilizer = hk.Linear(1, name="next_stabilizer")

        def step(state: jax.Array, inputs: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            emb, ev, prev = inputs
            state = (state + emb) / math.sqrt(2.0)
            indicators = event_indicators(ev, prev) if c.attention_bias else None
            for layer in layers:
                state = layer(state, bias, indicators)
            return state, next_stabilizer(state)[..., 0].astype(jnp.float32)

        initial = jnp.zeros((b, s, c.width), dtype=dtype)
        state, auxiliary = hk.scan(step, initial, tuple(jnp.swapaxes(x, 0, 1) for x in (embedded, events, previous)))
        grid = scatter_grid(state, self.positions, c.distance)
        data = hk.Conv2D(c.width, 2, padding="VALID", name="readout_to_data")(grid)
        data = hk.Linear(c.readout_width, name="readout_project")(data)
        cycle = jnp.full((b, 1, 1, 1), jnp.log1p(t - 1), dtype=dtype)
        data = data + hk.Linear(c.readout_width, name="cycle_embed")(cycle)
        # The adapter defines logical_axis as the path's FIXED coordinate.
        # Varying that coordinate moves perpendicular to the logical path.
        pooled = perpendicular_pool(data, self.logical_axis)
        pooled = ResidualMLP(c.readout_width, c.readout_layers, c.readout_widening,
                             c.layer_norm_epsilon, name="readout_resnet")(pooled)
        logits = hk.Linear(1, name="readout_logits")(pooled)[..., 0].astype(jnp.float32)
        return {"logits": logits[:, self.logical_index], "all_logits": logits,
                "next_stabilizer_logits": jnp.swapaxes(auxiliary, 0, 1)}


def transformed_model(config: AlphaQubitConfig, geometry: dict[str, Any]) -> hk.Transformed:
    def forward(features: dict[str, jax.Array]) -> dict[str, jax.Array]:
        dtype = jnp.bfloat16 if config.dtype == "bfloat16" else jnp.float32
        policy = jmp.Policy(param_dtype=jnp.float32, compute_dtype=dtype, output_dtype=dtype)
        with hk.mixed_precision.push_policy(hk.Linear, policy), \
             hk.mixed_precision.push_policy(hk.Conv2D, policy), \
             hk.mixed_precision.push_policy(hk.Embed, policy):
            return AlphaQubit(config, np.asarray(geometry["stabilizer_positions"]),
                              np.asarray(geometry["stabilizer_types"]),
                              geometry["logical_axis"], geometry["logical_index"])(features)
    return hk.without_apply_rng(hk.transform(forward))


def loss_terms(outputs: dict[str, jax.Array], features: dict[str, jax.Array],
               labels: jax.Array, auxiliary_weight: float = 0.02) -> tuple[jax.Array, dict[str, jax.Array]]:
    import optax

    labels = jnp.asarray(labels, dtype=jnp.float32).reshape(-1)
    main = optax.sigmoid_binary_cross_entropy(outputs["logits"], labels).mean()
    # Predict the next *measured bulk* stabilizers, never the inferred final
    # boundary: those are not actual stabilizer measurements.
    target = features["measurements"][:, 1:-1, :].astype(jnp.float32)
    predicted = outputs["next_stabilizer_logits"][:, :-2, :]
    aux = (optax.sigmoid_binary_cross_entropy(predicted, target).mean()
           if target.shape[1] else jnp.asarray(0.0, dtype=jnp.float32))
    total = main + auxiliary_weight * aux
    return total, {"loss": total, "main_loss": main, "aux_loss": aux,
                   "main_error": jnp.mean((outputs["logits"] > 0) != labels)}
