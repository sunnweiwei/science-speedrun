"""Check checkpoint wrapping preserves stochastic outputs, gradients and keys."""

import argparse
import os
import tempfile
import copy
import json
import sys
from pathlib import Path
import torch
from olmoearth_pretrain.nn.attention import Block

parser = argparse.ArgumentParser()
parser.add_argument("--fsdp", action="store_true")
parser.add_argument("--cross-attention", action="store_true")
parser.add_argument("--keyword-input", action="store_true")
args = parser.parse_args()
if args.fsdp:
    if (
        os.environ.get("EARTH_SLURM_PARTITION")
        and os.environ.get("SLURM_JOB_PARTITION") != os.environ["EARTH_SLURM_PARTITION"]
    ):
        raise RuntimeError("CUDA tests require configured-gpu-partition")
    from torch.distributed.fsdp import MixedPrecisionPolicy

    torch.cuda.set_device(0)
    rendezvous = tempfile.TemporaryDirectory()
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{rendezvous.name}/init", rank=0, world_size=1
    )
torch.set_num_threads(4)
torch.manual_seed(42)
block = Block(dim=64, num_heads=4, drop_path=0.1, cross_attn=args.cross_attention)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/baseline/starter"))
from recomputation import wrap_block

wrapped = wrap_block(copy.deepcopy(block))
assert list(block.state_dict()) == list(wrapped.state_dict())
x = torch.randn(3, 8, 64)
if args.fsdp:
    x = x.cuda()
    for module in (block, wrapped):
        module.cuda()
        module.apply_fsdp(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            )
        )
context = {"y": torch.randn(3, 6, 64, device=x.device)} if args.cross_attention else {}
outputs, gradients = [], []
for module in (block, wrapped):
    torch.manual_seed(123)
    output = module(x=x, **context) if args.keyword_input else module(x, **context)
    output.square().mean().backward()
    outputs.append(output.detach())
    gradients.append(
        torch.cat(
            [
                (p.grad.full_tensor() if args.fsdp else p.grad).flatten()
                for p in module.parameters()
            ]
        )
    )
assert torch.equal(outputs[0], outputs[1])
assert torch.equal(gradients[0], gradients[1])
print(
    json.dumps(
        {
            "fsdp": args.fsdp,
            "cross_attention": args.cross_attention,
            "keyword_input": args.keyword_input,
            "output_max_abs": float((outputs[0] - outputs[1]).abs().max()),
            "gradient_max_abs": float((gradients[0] - gradients[1]).abs().max()),
            "state_dict_keys_equal": True,
        }
    )
)

if args.fsdp:
    torch.distributed.destroy_process_group()
    rendezvous.cleanup()
