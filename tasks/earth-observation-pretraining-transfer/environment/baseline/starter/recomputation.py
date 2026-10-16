"""Checkpoint official blocks with FSDP input casting outside recomputation."""

from types import MethodType
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint

RECOMPUTATION_VERSION = 2


def checkpoint_including_keyword_tensors(function, *args, **kwargs):
    # PyTorch 2.9 infers CUDA devices for RNG preservation from positional args.
    # Official blocks are called with x=..., so passing those kwargs directly
    # misses CUDA RNG and changes DropPath masks during backward recomputation.
    # Positional containers expose every tensor to its pytree device traversal.
    def call(packed_args, packed_kwargs):
        return function(*packed_args, **packed_kwargs)

    return checkpoint(call, args, kwargs, use_reentrant=False, preserve_rng_state=True)


def shard_checkpoint_wrapper(self, **kwargs):
    # Delegating to the inner block's bound apply_fsdp would cast inputs only
    # after checkpoint captures them; backward recomputation then sees fp32
    # inputs with bf16 weights. Shard the wrapper instead so inputs are captured
    # after FSDP casting. Parameter names are stripped by CheckpointWrapper.
    fully_shard(self, **kwargs)


def wrap_block(block):
    wrapped = checkpoint_wrapper(
        block, checkpoint_fn=checkpoint_including_keyword_tensors
    )
    wrapped.apply_fsdp = MethodType(shard_checkpoint_wrapper, wrapped)
    return wrapped
