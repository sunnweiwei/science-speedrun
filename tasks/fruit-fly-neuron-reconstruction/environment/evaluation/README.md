# Development evaluation

Use only the visible validation image and reference suite under
`/workspace/data/validation`. `evaluate_physical.py --help` describes scoring
an EM reconstruction on that suite. Prediction metadata must use the actual
physical grid; never change labels or geometry to improve a score.

The task fixes one FFN checkpoint, ascending seeds, no consensus, shared SENSE
proposals, and disjoint two-node SHAPE merges. The repository implementation
`single_pipeline.py` executes these prediction stages from an image and a
model package. It does not take a reference suite as input. Its container work
root is `/workspace/research/run`; use a fresh root for each evaluation.

The independent private verifier runs the same prediction stages in a separate
namespace, then scores all three methods. Its primary reward is SHAPE nERL;
FFN-only and shared-proposal FFN remain controls. Runtime/partial-execution
failures are not evidence of low biological quality.

## Runtime of a full development evaluation

The validation image has the same 1794³ shape as the private test image, so a
complete `single_pipeline.py` run costs about as much as the formal verifier.
The pipeline uses every visible GPU:

- seeds for all 125 FFN windows are computed once on all CPUs
  (`precompute_seeds.py`, a few minutes);
- FFN runs four workers per GPU under a private CUDA MPS daemon when
  `nvidia-cuda-mps-control` is available, with the exact `norm32` LayerNorm
  kernel (about 925 33³ views per second per H100; `--inference-backend eager`
  gives about 460);
- SENSE, FFN-pair and SHAPE run four shards per GPU.

FFN cost depends on the checkpoint. A baseline-recipe checkpoint made 53 million
FFN calls over the 125 windows, which is about two hours on eight H100s, and 20
of its windows reached the 1,000,000-call cap described in the contract. Use
smaller window or region checks for quick feedback and full runs to confirm.
