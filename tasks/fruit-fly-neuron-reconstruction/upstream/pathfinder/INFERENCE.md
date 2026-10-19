# FFN inference acceleration

`launch_overlap.py` and `infer_overlap.py worker` accept
`--inference-backend norm32` (specialized LayerNorm) or
`--inference-backend norm32-graph` (also CUDA graphs). The default is `norm32`
on every CUDA GPU, including H100: no GPU-model selection or automatic fallback.
Startup parity, dependency and runtime errors fail explicitly. Use
`--inference-backend eager` to explicitly select the original implementation.
Training, checkpoint parameters, growth rules, seed order, and scoring are
unchanged. Shard receipts record the backend; merging different backends is
rejected. Existing receipts without this field mean `eager`.

The optional backend specializes the 32-channel LayerNorm used by FFN15 and
captures CUDA graphs for each actual batch size. It preserves the reference
Welford operation order and does not pad the final batch. CUDA graph capture
keeps autocast weight casts inside the graph so temporary warmup tensors cannot
become dangling graph inputs. Returned NumPy outputs remain valid across replay.

The pinned runtime is PyTorch 2.6.0 and Triton 3.2.0. Container recipes include
GCC, libc headers and setuptools 75.8.0 for Triton's runtime compilation.
Each model receives a LayerNorm parity check before replacement. This finite
startup check is not a proof for arbitrary inputs or checkpoints. Measurements
were made on A5000; H100 is enabled by default but has not been measured or
validated end to end here. Removing the device gate does not change the kernel
arithmetic used for the recorded A5000 measurements.

## Measurement scope

Measurements use one fixed checkpoint, 33-cubed network inputs, BF16 autocast,
and one RTX A5000. Whole-window validation uses the same 500-cubed validation
window, seeds, and raw data as the eager reference. It is not full-volume
verifier timing and does not include SENSE/SHAPE.

The eager operator profile assigns 45.26% of self CUDA time to LayerNorm and
32.72% to convolution. A 128-input replay measured 154.10 FOV/s for eager,
163.77 FOV/s with CUDA graphs alone, and 267.40 FOV/s with the specialized
LayerNorm plus graphs. The latter replay matched every reference output exactly.
These short measurements do not establish end-to-end speedup.

Increasing eager batch size from 1 to 16 only raised throughput from 154.10 to
165.07 FOV/s. Device memory usage was approximately 289 MiB at batch 1 and
667 MiB at batch 16, out of 24564 MiB. PyTorch peak allocated tensor memory was
26.74 and 303.28 MiB respectively. Memory capacity is not the current bottleneck.
`nvidia-smi` memory utilization measures memory-controller activity; it is not
allocated capacity or a measurement of percentage of peak bandwidth.

General `torch.compile` was also tested: full-window elapsed time fell from
3158.33 to 1607.99 seconds, but local nERL fell from 0.0963434 to 0.0839459.
It is not used by this backend. Small changes to logits can affect subsequent
FFN growth, so forward throughput alone is insufficient validation.

## Full-window result

The specialized backend completed 420,448 predictions in **2002.91 seconds
(33.38 minutes)**, compared with **3158.33 seconds (52.64 minutes)** for eager:
**1.577x faster**, or 36.6% less elapsed time. Prediction time fell from 2694.50
to 1572.60 seconds. Setup is measured separately from both window timings.
The entire 500-cubed segmentation array matched exactly, prediction counts
matched, and local nERL remained **0.09634338594047565**.

Across 1,898 samples during the window, device memory stayed at **321 MiB of
24564 MiB (1.31%)**. GPU busy had median **83%** and memory-controller busy
median **31%**. These busy percentages describe activity, not achieved fractions
of peak FLOPS or peak bandwidth. Sampling includes the CPU seed-selection
phase. CUDA graph memory remains reserved across replay; PyTorch peaks reset
after capture should not be interpreted as the whole backend's memory footprint.

See the [machine-readable receipt](receipts/ffn-inference-a5000.json) for hashes,
metrics, test coverage, and artifact locations. This establishes parity for the
measured checkpoint/window, not all checkpoints or GPU architectures.

## Tests

CPU contract and receipt tests:

```sh
python -m unittest test_ffn_backend test_ffn_inference test_infer_overlap test_launch_overlap test_single_pipeline
```

Within a registered A5000 allocation with the supported runtime:

```sh
PATHFINDER_RUN_GPU_TESTS=1 python -m unittest test_ffn_backend test_ffn_inference
```

The GPU test checks batches 1, 2, 3, and 4 against eager, repeated graph replay,
output lifetime, and unchanged checkpoint state. GPU tests require an explicit
Slurm allocation and are skipped in the default CPU run.

## Combining the backend with PR #150's MPS approach (A5000)

A controlled one-GPU experiment compares one worker, four workers without MPS,
and four workers with MPS, each using batch 4. Workers execute 128 real recurrent
FFN growth steps for each of four independent copies of validation seed 81.
Each configuration is repeated three times; the table reports median aggregate
FOV/s. Setup and initial seed selection are excluded. MPS server/client queries
confirmed that all four clients were attached to the allocated GPU's MPS server.

| Worker configuration | Eager | Specialized LayerNorm | LayerNorm + graph |
| --- | ---: | ---: | ---: |
| One worker | 153.28 | 224.49 | 232.10 |
| Four workers, no MPS | 156.45 | 245.64 | 249.73 |
| Four workers, MPS | 173.57 | **278.60** | 275.14 |

LayerNorm adds **1.605x throughput** over the MPS eager baseline (37.7% less
steady-state time per fixed amount of work). Relative to one eager worker, the
combination reaches 1.82x throughput. Graph capture adds no benefit in the
measured MPS configuration; use `--inference-backend norm32` when integrating
with the MPS launcher. The full MPS pipeline also accepts `single_pipeline.py --ffn-inference-backend
norm32` together with its usual `--root`, `--runtime`, and `--limits` arguments.
The verifier defaults to `norm32` on every CUDA GPU. Worker receipts record both
requested and actual backends. CUDA Graph remains an explicit option, with no
automatic fallback or device-based dispatch.

All prediction trace hashes matched eager in every repeat and configuration.
This is a repeated real seed trace, not a full-volume or multi-region quality
validation. The earlier whole-window result remains the separate evidence for
single-worker `norm32-graph` parity. No H100 result is inferred here.

Four-worker MPS median device memory was 1449 MiB for eager, 1377 MiB for
LayerNorm, and 1481 MiB with graphs, on a 24564 MiB A5000. GPU busy median was
100% in all three; memory-controller busy medians were 38%, 60%, and 59%.
Memory capacity remains ample. GPU busy at 100% does not imply peak compute
throughput, as the performance differences demonstrate.

See [MPS measurement receipt](receipts/ffn-inference-mps-a5000.json) for all
three repeats, memory statistics, provenance and scope limits.
