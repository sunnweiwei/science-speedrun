#!/usr/bin/env python3
"""Bounded full-resolution WN2 forward, LoRA and from-scratch training probes.

Uses public demo data only. This does not establish forecast skill or change v1.
"""
import argparse
import copy
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
import xarray_jax
from weathernext.utils import casting, checkpoint, data_utils, fiddle_config_io, sparse_transformer, xarray_tree
from weathernext.weathernext2 import fgn

LORA_MODULE = re.compile(r"/block_\d+/(mha_proj_[qkv]|mha_final|ffw_up|ffw_down)$")

_DETERMINISTIC_GRAPH_OPS = False
_SERIAL_ENSEMBLE = False


def ensure_deterministic_process():
    """Set Python hashing before interpreter startup, including Slurm resumes."""
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  os.environ | {'PYTHONHASHSEED': '0'})

def enable_serial_ensemble():
    """Serialize native sample/batch network calls after native noise creation.

    The original FGN method still creates the complete noise tensor and computes
    the same two-member predictions. Only the expensive neural network is
    scanned one member at a time; CRPS and autoregressive feedback are unchanged.
    """
    global _SERIAL_ENSEMBLE
    if _SERIAL_ENSEMBLE:
        return
    import copy
    original = fgn.Predictor.run_model_with_sample_as_batch

    def serial(self, *args, **kwargs):
        instance = copy.copy(self)
        network = self._noisy_function

        def call(*, inputs, targets_template, forcings, is_training):
            if inputs.sizes['batch'] == 1:
                return network(inputs=inputs, targets_template=targets_template,
                               forcings=forcings, is_training=is_training)
            datasets = (inputs, targets_template, forcings)
            scans, definitions = [], []
            for dataset in datasets:
                if dataset is None:
                    scans.append(None)
                    definitions.append(None)
                    continue
                dataset = dataset.transpose('batch', ...)
                if 'batch' in dataset.indexes:
                    dataset = dataset.drop_indexes(list(dataset.indexes['batch'].names) + ['batch']) if hasattr(dataset.indexes['batch'], 'levels') else dataset.drop_indexes('batch')
                dataset = dataset.drop_vars([n for n in ('batch', '_batch', 'sample') if n in dataset.coords])
                leaves, _ = jax.tree_util.tree_flatten(dataset)
                _, definition = jax.tree_util.tree_flatten(dataset.isel(batch=slice(0, 1)))
                scans.append([a[:, None, ...] for a in leaves])
                definitions.append(definition)
            output_definition = []

            def step(carry, arrays):
                values = [None if d is None else jax.tree_util.tree_unflatten(d, a)
                          for d, a in zip(definitions, arrays)]
                result = network(inputs=values[0], targets_template=values[1],
                                 forcings=values[2], is_training=is_training)
                leaves, definition = jax.tree_util.tree_flatten(result)
                output_definition[:] = [definition]
                return carry, leaves

            _, results = hk.scan(hk.remat(step), (), tuple(scans))
            result = jax.tree_util.tree_unflatten(output_definition[0], [a[:, 0, ...] for a in results])
            # Restore the original combined batch/sample index for native unstack.
            return result.assign_coords(xr.Coordinates.from_pandas_multiindex(inputs.indexes['batch'], 'batch'))

        instance._noisy_function = call
        return original(instance, *args, **kwargs)

    fgn.Predictor.run_model_with_sample_as_batch = serial
    _SERIAL_ENSEMBLE = True


def enable_deterministic_graph_ops():
    """Keep native graph sums, using fixed sparse reductions without atomics.

    Bucket static receiver rows by degree and gather only their actual edges.
    This avoids the large generic deterministic-scatter workspace on GPU.
    Padding indices are dropped, as in native segment_sum. The adjoint is the
    exact corresponding gather. No graph edges, layers or parameters change.
    """
    global _DETERMINISTIC_GRAPH_OPS
    if _DETERMINISTIC_GRAPH_OPS:
        return
    from weathernext.utils import gather_scatter_ops, sharding

    def factory(indices, num_output_rows, indices_are_sorted=False,
                check_is_sorted=False, explicit_sharding=True,
                remove_collectives_if_local=True, _use_custom_vjp=True):
        del explicit_sharding, remove_collectives_if_local, _use_custom_vjp
        if sharding.is_global_mesh_defined():
            raise ValueError('Task graph sums support replicated data parallelism only')
        indices = np.asarray(indices, dtype=np.int32)
        if indices.ndim != 1:
            raise ValueError('Static graph indices must be one-dimensional')
        if indices_are_sorted and check_is_sorted and not gather_scatter_ops.is_sorted(indices):
            raise ValueError('Native sorted graph-index contract violated')
        valid = (indices >= 0) & (indices < num_output_rows)
        positions = np.flatnonzero(valid)
        positions = positions[np.argsort(indices[positions], kind='stable')]
        counts = np.bincount(indices[positions], minlength=num_output_rows)
        starts = np.cumsum(counts) - counts
        groups = []
        row_order = []
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            groups.append((len(empty), None))
            row_order.append(empty)
        width = 1
        while width <= max(1, int(counts.max(initial=0))) * 2:
            rows = np.flatnonzero((counts > width // 2) & (counts <= width))
            if len(rows):
                offsets = np.arange(width)[None, :]
                slots = starts[rows, None] + offsets
                gathered = positions[np.minimum(slots, len(positions)-1)]
                gathered = np.where(offsets < counts[rows, None], gathered, -1).astype(np.int32)
                groups.append((len(rows), gathered))
                row_order.append(rows)
            width *= 2
        inverse = np.argsort(np.concatenate(row_order)).astype(np.int32)

        @jax.custom_vjp
        def reduce(x):
            values = []
            for rows, gather_indices in groups:
                if gather_indices is None:
                    values.append(jnp.zeros((rows, *x.shape[1:]), dtype=x.dtype))
                else:
                    taken = gather_scatter_ops.gather_with_fill(x, jnp.asarray(gather_indices.ravel()))
                    taken = taken.reshape((rows, gather_indices.shape[1], *x.shape[1:]))
                    # Accumulate BF16 messages in FP32, then restore native
                    # activation dtype. This is still the same unweighted sum.
                    values.append(jnp.sum(taken, axis=1, dtype=jnp.float32).astype(x.dtype))
            return jnp.concatenate(values, axis=0)[inverse]

        def forward(x):
            return reduce(x), None

        def backward(_, cotangent):
            return (gather_scatter_ops.gather_with_fill(cotangent, jnp.asarray(indices)),)

        reduce.defvjp(forward, backward)

        def call(x, segment_ids=None, num_segments=None):
            if isinstance(segment_ids, np.ndarray):
                np.testing.assert_array_equal(segment_ids, indices)
            if isinstance(num_segments, int) and num_segments != num_output_rows:
                raise ValueError('Native graph row count changed')
            return reduce(x)
        return call

    gather_scatter_ops.create_static_segment_sum = factory
    _DETERMINISTIC_GRAPH_OPS = True


def enable_transformer_remat():
    """Recompute block activations during backward, preserving Haiku names."""
    original = sparse_transformer.Block.__call__

    def call(self, x, global_norm_conditioning):
        return hk.remat(lambda value, conditioning: original(
            self, value, global_norm_conditioning=conditioning))(x, global_norm_conditioning)

    sparse_transformer.Block.__call__ = call


def eligible(params):
    return [(module, name, value.shape) for module, entries in params.items()
            for name, value in entries.items()
            if LORA_MODULE.search(module) and name == "w" and value.ndim == 2]


def construct_predictor(config, precision):
    """Cast only the neural network; keep normalization and CRPS in float32.

    xarray Dataset.mean(numeric_only=True) drops bfloat16 variables in this
    runtime. Casting the complete loss wrapper therefore silently produces an
    empty dataset in FGN's target-centering operation. The official FGN loss
    stays unchanged; its inputs and reductions retain float32 precision.
    """
    enable_deterministic_graph_ops()
    enable_serial_ensemble()
    predictor_kwargs = dict(config.predictor_kwargs)
    noisy_kwargs = copy.deepcopy(predictor_kwargs['noisy_function_kwargs'])
    noisy_kwargs.update(remat_grid_to_mesh_gnn=True, remat_mesh_gnn=True,
                        remat_mesh_to_grid_gnn=True)
    def remat_dense(value):
        if isinstance(value, functools.partial):
            remat_dense(value.keywords)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key.endswith('dense_kwargs') and isinstance(child, dict):
                    # Upstream DenseLayer supports rematerializing its MLP and
                    # normalization together, without changing weights/layers.
                    child['remat'] = True
                remat_dense(child)
        elif isinstance(value, (tuple, list)):
            for child in value:remat_dense(child)
    remat_dense(noisy_kwargs)
    predictor_kwargs['noisy_function_kwargs'] = noisy_kwargs
    config = dataclasses.replace(config, predictor_kwargs=predictor_kwargs)
    if precision == "bfloat16":
        kwargs = dict(config.predictor_kwargs)
        constructor = kwargs["noisy_function_constructor"]
        def mixed_network(**kw):
            network = constructor(**kw)

            # Bfloat16Cast forwards three positional arguments, whereas the
            # released ForwardPass accepts keyword-only arguments. A plain
            # function adapter preserves the original Haiku module names.
            def keyword_call(inputs, targets_template, forcings, **extra):
                return network(inputs=inputs, targets_template=targets_template,
                               forcings=forcings, **extra)

            return casting.Bfloat16Cast(keyword_call)

        kwargs["noisy_function_constructor"] = mixed_network
        config = dataclasses.replace(config, predictor_kwargs=kwargs)
    return fgn.construct_predictor(config)


def main():
    ensure_deterministic_process()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=["wn2", "cyclones"], default="wn2")
    parser.add_argument("--mode", choices=["audit", "forward", "lora", "random-train"], default="audit")
    parser.add_argument("--rank", type=int, default=148)
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--remat-transformer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--distributed", action="store_true", help="Four-GPU accumulation/resume probe")
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.mode == "random-train" and args.model != "wn2":
        parser.error("The from-scratch route uses WN2, not the cyclone variant")
    if (args.distributed or args.resume) and args.mode != "random-train":
        parser.error("Distributed/resume options require random-train")
    if args.resume and not args.distributed:
        parser.error("Resume requires the distributed probe")
    if not 1 <= args.accumulation <= 16:
        parser.error("accumulation must be 1..16")
    if not 1 <= args.repeats <= 3 or not 1 <= args.rank <= 768:
        parser.error("repeats must be 1..3 and rank must be 1..768")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "mode": args.mode, "status": "started",
              "scientific_skill_evaluation": False,
              "pretrained_parameters_loaded": args.mode != "random-train",
              "sample_after_pretraining_cutoff": args.model == "cyclones",
              "source_commit": "89c4b2a77a1c57b328b909c575550fd2e5aadc9c",
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "precision": args.precision, "attention": "triblockdiag_mha"}
    report["xla_flags"] = os.environ.get("XLA_FLAGS", "")

    def save(**updates):
        report.update(updates)
        tmp = args.output / "report.json.tmp"
        tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        tmp.replace(args.output / "report.json")
        print(json.dumps(updates, default=str), flush=True)

    try:
        source_root = Path(fgn.__file__).resolve().parents[2]
        actual_revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        if actual_revision != report["source_commit"]:
            raise ValueError(f"Unexpected WeatherNext source revision: {actual_revision}")
        report["versions"] = {name: importlib.metadata.version(name)
                              for name in ["jax", "jaxlib", "dm-haiku", "optax", "numpy", "xarray"]}
        manifest = json.loads((args.assets / "manifest.json").read_text())
        required_files = ["sample.nc"]
        if args.mode != "random-train":
            required_files.append(args.model + ".npz")
        for filename in required_files:
            entry = next(a for a in manifest["assets"] if a["file"] == filename)
            digest = hashlib.sha256()
            with (args.assets / filename).open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Asset hash mismatch: {filename}")
        report["asset_manifest"] = manifest
        params = None
        if args.mode != "random-train":
            with (args.assets / (args.model + ".npz")).open("rb") as handle:
                ckpt = checkpoint.load(handle, fgn.CheckPoint)
            params = ckpt.params
            matrices = eligible(params)
            base_count = sum(a.size for a in jax.tree_util.tree_leaves(params))
            adapter_count = args.rank * sum(sum(shape) for _, _, shape in matrices)
            if len(matrices) != 144:
                raise ValueError(f"Expected 24 x 6 LoRA matrices, got {len(matrices)}")
            save(base_parameters=base_count, base_parameter_bytes=sum(a.nbytes for a in jax.tree_util.tree_leaves(params)),
                 lora={"rank": args.rank, "matrices": len(matrices), "parameters": adapter_count,
                       "protein_trainable_parameters": 49198221,
                       "relative_difference_from_protein": adapter_count / 49198221 - 1,
                       "modules": [m for m, _, _ in matrices]}, checkpoint_description=ckpt.description)
        config_name = "WeatherNext2" if args.model == "wn2" else "WeatherNextCyclones"
        config = fiddle_config_io.get_fiddle_config_by_name("weathernext2/configs/" + config_name)
        noisy = config.predictor_kwargs["noisy_function_kwargs"]
        transformer = noisy["mesh_model_ctor"].keywords["transformer_kwargs"]
        transformer["attention_type"] = "triblockdiag_mha"
        dataset = xr.load_dataset(args.assets / "sample.nc")
        if args.model == "wn2":
            winds_info = json.loads((args.assets / "winds.json").read_text())
            if hashlib.sha256((args.assets / "winds.nc").read_bytes()).hexdigest() != winds_info["sha256"]:
                raise ValueError("ERA5 100m wind fixture checksum mismatch")
            winds = xr.load_dataset(args.assets / "winds.nc")
            np.testing.assert_array_equal(winds.datetime.values, dataset.datetime.values)
            dataset = xr.merge([dataset, winds], join="exact", compat="equals")
            save(ancillary_winds=winds_info, mixed_hres_era5_execution_fixture=True)
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            dataset, target_lead_times=slice("6h", "6h"), **dataclasses.asdict(config.task))
        if inputs.sizes["lat"] != 721 or inputs.sizes["lon"] != 1440:
            raise ValueError("Full WN2 validation requires native 0.25-degree input")
        save(architecture={"layers": transformer["num_layers"], "width": transformer["d_model"],
                           "heads": transformer["num_heads"], "mesh_splits": noisy["mesh_num_splits"]},
             inputs={"sizes": dict(inputs.sizes), "variables": list(inputs.data_vars)},
             targets={"sizes": dict(targets.sizes), "variables": list(targets.data_vars)},
             training_ensemble_members=config.predictor_wrappers[-1]["kwargs"]["num_samples"])
        if args.mode == "audit":
            save(status="passed")
            return
        if jax.default_backend() != "gpu":
            raise RuntimeError("Execution validation requires an allocated GPU")
        training = args.mode in ("lora", "random-train")
        if training and args.remat_transformer:
            enable_transformer_remat()
        report["transformer_block_rematerialization"] = training and args.remat_transformer
        save(devices=[{"device": str(d), "kind": d.device_kind} for d in jax.devices()], phase="constructing")
        inference_config = dataclasses.replace(config, predictor_wrappers=config.predictor_wrappers[:-1])

        def predictor(selected_config):
            return construct_predictor(selected_config, args.precision)

        @hk.transform
        def forward(i, template, f):
            return predictor(inference_config)(i, targets_template=template, forcings=f)

        @hk.transform
        def loss(i, target, f):
            value, diagnostics = predictor(config).loss(i, target, f)
            return xarray_tree.map_structure(
                lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True), (value, diagnostics))

        if args.mode == "random-train":
            save(phase="random_initialization", initialization_seed=0,
                 initialization="Native Haiku initializers; no checkpoint parameter arrays loaded",
                 normalization="Released config statistics, for execution probing only; not audited for a research split")
            start = time.monotonic()
            params = jax.jit(forward.init)(
                jax.random.PRNGKey(0), inputs, xr.zeros_like(targets), forcings)
            jax.block_until_ready(params)
            base_count = sum(a.size for a in jax.tree_util.tree_leaves(params))
            if base_count != 183785575:
                raise ValueError(f"Unexpected full WN2 parameter count: {base_count}")
            save(initialization_seconds=time.monotonic() - start,
                 base_parameters=base_count, trainable_parameters=base_count)
        else:
            params = jax.device_put(params)
        if args.mode == "forward":
            run = jax.jit(forward.apply)
            # No target values enter inference, including future cyclone labels.
            template = xr.zeros_like(targets)
            save(phase="forward_compile")
            start = time.monotonic()
            run = run.lower(params, jax.random.PRNGKey(42), inputs, template, forcings).compile()
            save(compile_seconds=time.monotonic() - start,
                 xla_cost_analysis=run.cost_analysis(), xla_memory_analysis=str(run.memory_analysis()))
            durations = []
            for iteration in range(args.repeats):
                save(phase="forward", iteration=iteration)
                start = time.monotonic()
                prediction = run(params, jax.random.PRNGKey(42), inputs, template, forcings)
                jax.block_until_ready(prediction)
                durations.append(time.monotonic() - start)
                checks = {}
                for name in targets:
                    valid = np.isfinite(targets[name].values)
                    values = np.asarray(prediction[name].transpose(*targets[name].dims).data)
                    checks[name] = {"shape": list(values.shape), "valid_target_values": int(valid.sum()),
                                    "finite_at_valid_targets": bool(np.isfinite(values[valid]).all())}
                if not all(c["finite_at_valid_targets"] for c in checks.values()):
                    raise ValueError("Nonfinite forecast at valid target locations")
                save(seconds=durations, prediction_checks=checks, memory=jax.devices()[0].memory_stats())
                del prediction
        elif args.mode == "random-train" and args.distributed:
            from wn2_distributed import run_probe
            run_probe(params, loss, inputs, targets, forcings, args.output, save,
                      accumulation=args.accumulation, resume=args.resume)
        elif args.mode == "random-train":
            # One-step native WN2 all-parameter training. The short probe uses
            # a fixed small LR; it is not a compressed pretraining schedule.
            optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                                    optax.adamw(1e-5, b2=0.95, weight_decay=0.1))
            state = optimizer.init(params)
            save(optimizer={"name": "AdamW", "learning_rate": 1e-5,
                            "beta2": 0.95, "weight_decay": 0.1,
                            "gradient_clip": 1.0, "scope": "two-update execution probe"})

            @jax.jit
            def step(p, opt_state, rng, i, t, f):
                (value, _), gradient = jax.value_and_grad(loss.apply, has_aux=True)(p, rng, i, t, f)
                updates, opt_state = optimizer.update(gradient, opt_state, p)
                finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a))
                                           for a in jax.tree_util.tree_leaves(gradient)]))
                updated = optax.apply_updates(p, updates)
                changed = jnp.any(jnp.stack([jnp.any(a != b) for a, b in zip(
                    jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(updated))]))
                params_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a))
                                                  for a in jax.tree_util.tree_leaves(updated)]))
                return updated, opt_state, value, optax.global_norm(gradient), finite & params_finite, changed

            save(phase="random_train_compile")
            start = time.monotonic()
            step = step.lower(params, state, jax.random.PRNGKey(1), inputs, targets, forcings).compile()
            save(compile_seconds=time.monotonic() - start,
                 xla_cost_analysis=step.cost_analysis(), xla_memory_analysis=str(step.memory_analysis()))
            durations, losses = [], []
            for iteration in range(args.repeats):
                save(phase="random_train_update", iteration=iteration)
                start = time.monotonic()
                params, state, value, norm, finite, changed = step(
                    params, state, jax.random.PRNGKey(iteration + 1), inputs, targets, forcings)
                jax.block_until_ready((params, state, value, norm, finite, changed))
                durations.append(time.monotonic() - start)
                if not bool(finite) or not math.isfinite(float(value)) or not float(norm) > 0 or not bool(changed):
                    raise ValueError("Random-training loss/gradient/parameter-update gate failed")
                losses.append(float(value))
                save(seconds=durations, losses=losses, gradient_norm=float(norm),
                     parameters_changed=bool(changed), memory=jax.devices()[0].memory_stats())
            with (args.output / "probe-weights.npz").open("wb") as handle:
                checkpoint.dump(handle, fgn.CheckPoint(
                    params=jax.device_get(params), description="WN2 random-init execution probe; not a scientific starter",
                    license="Apache-2.0 code; see upstream data and configuration terms"))
        else:
            random = np.random.default_rng(0)
            adapters = {module: {"a": jnp.asarray(random.normal(0, 0.01, (shape[0], args.rank)).astype(np.float32)),
                                 "b": jnp.zeros((args.rank, shape[1]), dtype=jnp.float32)}
                        for module, _, shape in matrices}
            optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-6))
            state = optimizer.init(adapters)

            def objective(adapter, base, rng, i, t, f):
                merged = {module: dict(entries) for module, entries in base.items()}
                for module, name, _ in matrices:
                    merged[module][name] = jax.lax.stop_gradient(base[module][name]) + adapter[module]["a"] @ adapter[module]["b"]
                return loss.apply(merged, rng, i, t, f)

            @jax.jit
            def step(adapter, opt_state, base, rng, i, t, f):
                (value, diagnostics), gradient = jax.value_and_grad(objective, has_aux=True)(adapter, base, rng, i, t, f)
                updates, opt_state = optimizer.update(gradient, opt_state, adapter)
                finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree_util.tree_leaves(gradient)]))
                return optax.apply_updates(adapter, updates), opt_state, value, diagnostics, optax.global_norm(gradient), finite

            save(phase="lora_compile")
            start = time.monotonic()
            step = step.lower(adapters, state, params, jax.random.PRNGKey(0), inputs, targets, forcings).compile()
            save(compile_seconds=time.monotonic() - start,
                 xla_cost_analysis=step.cost_analysis(), xla_memory_analysis=str(step.memory_analysis()))
            durations, losses = [], []
            for iteration in range(args.repeats):
                save(phase="lora_backward_update", iteration=iteration)
                start = time.monotonic()
                adapters, state, value, diagnostics, norm, finite = step(
                    adapters, state, params, jax.random.PRNGKey(iteration), inputs, targets, forcings)
                jax.block_until_ready((adapters, state, value, norm, finite))
                durations.append(time.monotonic() - start)
                losses.append(float(value))
                if not bool(finite) or not math.isfinite(float(value)) or not float(norm) > 0:
                    raise ValueError("Loss/gradient gate failed")
                changed = any(bool(jnp.any(a["b"] != 0)) for a in adapters.values())
                if not changed:
                    raise ValueError("LoRA update did not change zero-initialized B")
                save(seconds=durations, losses=losses, gradient_norm=float(norm), adapters_changed=changed,
                     memory=jax.devices()[0].memory_stats())
        save(status="passed", phase="complete")
    except Exception as error:
        save(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
