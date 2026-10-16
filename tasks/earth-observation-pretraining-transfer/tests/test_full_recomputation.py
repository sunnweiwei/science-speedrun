"""Compare official Large loss and every gradient on a real masked small batch."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--math-sdp", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--microbatches", type=int, default=1)
    a = p.parse_args()
    if a.microbatches < 1:
        p.error("microbatches must be positive")
    if (
        os.environ.get("EARTH_SLURM_PARTITION")
        and os.environ.get("SLURM_JOB_PARTITION") != os.environ["EARTH_SLURM_PARTITION"]
    ):
        raise RuntimeError("CUDA regression requires configured-gpu-partition")
    torch.set_num_threads(4)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if a.math_sdp:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    with tempfile.TemporaryDirectory() as tmp:
        if world_size > 1:
            torch.distributed.init_process_group("nccl", init_method="env://")
        else:
            torch.distributed.init_process_group(
                "nccl", init_method=f"file://{tmp}/init", rank=0, world_size=1
            )
        sys.path.insert(0, str(a.upstream / "scripts/official/v1"))
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[1] / "environment/baseline/starter")
        )
        import large
        import script
        from recomputation import wrap_block

        common = script.build_common_components(
            str(a.upstream / "scripts/official/v1/large.py"),
            "dry_run",
            "regression",
            "local",
            [],
        )
        config = large.build_model_config(common)
        dataset_config = script.build_dataset_config(common)
        dataset_config.h5py_dir = str(a.data)
        dataset = dataset_config.build()
        loader_config = script.build_dataloader_config(common)
        loader_config.work_dir = tmp
        loader_config.global_batch_size = 2 * world_size
        loader_config.num_workers = 0
        loader_config.prefetch_factor = None
        loader_config.min_patch_size = loader_config.max_patch_size = 4
        loader_config.sampled_hw_p_list = [2]
        loader_config.max_timesteps = 2
        loader_config.token_budget = 512
        loader = loader_config.build(dataset)
        # Per-rank temporary directories are intentionally separate. Keep the
        # deterministic global index array in memory instead of a shared cache.
        loader.reshuffle(epoch=1, in_memory=True)
        patch_size, view_a, view_b = next(iter(loader))
        views = [
            view_a.to_device(torch.device("cuda")),
            view_b.to_device(torch.device("cuda")),
        ]
        module_config = script.build_train_module_config(common)
        base_loss = module_config.loss_config.build()
        contrastive_loss = module_config.contrastive_config.build()
        reference_gradients = {}
        losses = []
        comparisons = {}
        initialization_hashes = []
        for mode, recompute in [
            ("native", False),
            ("native_repeat", False),
            ("recomputed", True),
        ]:
            torch.manual_seed(12536)
            model = config.build()
            initial_hash = hashlib.sha256()
            for parameter in model.parameters():
                initial_hash.update(parameter.detach().numpy().tobytes())
            initialization_hashes.append(initial_hash.hexdigest())
            if recompute:
                for part in [model.encoder, model.decoder]:
                    for i, block in enumerate(part.blocks):
                        part.blocks[i] = wrap_block(block)
            model.cuda().train()
            model.apply_fsdp(
                param_dtype=torch.float32 if a.fp32 else torch.bfloat16,
                reduce_dtype=torch.float32,
            )
            torch.manual_seed(777 + rank)
            total_loss = 0.0
            for micro in range(a.microbatches):
                # Match the official train module FSDP microbatch context.
                model.set_is_last_backward(micro == a.microbatches - 1)
                per_view, pooled = [], []
                for original_view in views:
                    view = type(original_view)(
                        *[
                            value.clone() if isinstance(value, torch.Tensor) else value
                            for value in original_view
                        ]
                    )
                    output = model(view, patch_size=patch_size)
                    with torch.no_grad():
                        target = model.target_encoder(
                            view.unmask(),
                            patch_size=patch_size,
                            token_exit_cfg=module_config.token_exit_cfg,
                        )["tokens_and_masks"]
                    per_view.append(base_loss.compute(output[1], target))
                    pooled.append(output[2])
                loss = sum(per_view) / 2 + contrastive_loss.compute(*pooled)
                loss = loss / a.microbatches
                loss.backward()
                total_loss += float(loss.detach().cpu())
            losses.append(total_loss)
            print(mode, "loss", losses[-1], flush=True)
            names = set()
            differences = []
            diff_squared = reference_squared = 0.0
            gradient_max_abs = 0.0
            for name, param in model.named_parameters():
                name = name.replace("._checkpoint_wrapped_module", "")
                names.add(name)
                grad = (
                    None
                    if param.grad is None
                    else param.grad.full_tensor().detach().cpu()
                )
                if mode == "native":
                    reference_gradients[name] = grad
                else:
                    reference = reference_gradients[name]
                    if (reference is None) != (grad is None):
                        raise AssertionError(f"Gradient participation differs: {name}")
                    if grad is not None:
                        diff = float((grad - reference).abs().max())
                        gradient_max_abs = max(gradient_max_abs, diff)
                        delta = grad - reference
                        diff_squared += float(delta.double().square().sum())
                        reference_squared += float(reference.double().square().sum())
                        if not torch.equal(grad, reference):
                            differences.append(
                                {
                                    "name": name,
                                    "max_abs": diff,
                                    "reference_max_abs": float(reference.abs().max()),
                                }
                            )
            assert names == set(reference_gradients)
            if mode != "native":
                comparisons[mode] = {
                    "gradient_max_abs": gradient_max_abs,
                    "gradient_relative_l2": (
                        diff_squared / max(reference_squared, 1e-300)
                    )
                    ** 0.5,
                    "nonidentical_parameters": len(differences),
                    "largest_differences": sorted(
                        differences, key=lambda v: v["max_abs"], reverse=True
                    )[:20],
                }
            del model, loss, output, target, per_view, pooled, param, grad
            gc.collect()
            torch.cuda.empty_cache()
        exact = len(set(losses)) == 1 and all(
            c["nonidentical_parameters"] == 0 for c in comparisons.values()
        )
        assert len(set(initialization_hashes)) == 1, "Initializations differ"
        report = {
            "model": "official v1 Large, encoder and predictor",
            "real_source": str(a.data),
            "batch": 2 * world_size,
            "rank_microbatch": 2,
            "accumulation_microbatches": a.microbatches,
            "same_real_microbatch_reused": True,
            "patch_size": patch_size,
            "sampled_hw_p": 2,
            "max_timesteps": 2,
            "fsdp_world_size": world_size,
            "parameter_dtype": "float32" if a.fp32 else "bfloat16",
            "math_sdp": a.math_sdp,
            "initialization_hashes": initialization_hashes,
            "losses": losses,
            "comparisons": comparisons,
            "all_parameter_gradients_identical": exact,
            "limitation": "Small real masked batch; does not prove long-run learning quality.",
        }
        rank_reports = [None] * world_size
        torch.distributed.all_gather_object(
            rank_reports,
            {
                "rank": rank,
                "losses": losses,
                "comparisons": comparisons,
                "all_parameter_gradients_identical": exact,
            },
        )
        exact = all(row["all_parameter_gradients_identical"] for row in rank_reports)
        report["all_parameter_gradients_identical"] = exact
        report["rank_reports"] = rank_reports
        if rank == 0:
            a.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
        torch.distributed.destroy_process_group()
        if not a.diagnostic and not exact:
            raise AssertionError("Full model gradients differ; inspect written report")


if __name__ == "__main__":
    main()
