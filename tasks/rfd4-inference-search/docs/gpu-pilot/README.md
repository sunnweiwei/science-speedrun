# GPU conformance pilot

These runs exercise the released weights and an independent scoring path on RTX A6000 GPUs. The fixed fixtures serve different purposes. The unconditioned 250-residue example has a publisher-supplied generated structure. The unconditioned 100-residue example exercises native AF3-guided beam selection. A frozen, previously published small-molecule complex exercises the native final evaluation. The 1Y26 crystal has quantitative upstream AF3 regression bounds. The four-suite Figure 2 reference remains the [archived operating-point comparison](../../environment/references/overall/README.md).

## Measured outcomes

| Check | Local observation |
| --- | --- |
| Published 250-residue output on H200 vs fresh A6000 sample | Aligned Cα RMSD 0.119 Å, backbone RMSD 0.120 Å, sequence agreement 235/250. A local same-seed rerun reproduced all 2,039 serialized atom coordinates exactly. Changing the seed gave Cα RMSD 24.79 Å against the first local draw. |
| Independent AF3 and Foldseek duplicate control | All four records passed. The official sample and two seed-42 local copies clustered together. The seed-43 control formed the second cluster. |
| Released 100-residue AF3 beam | Two complete runs on separate A6000s, each with two replicates and 48 real internal AF3 evaluations. The observed top-three indices agreed with independent stable sorting at all eight checkpoints in the instrumented run. The ordered selections also agreed across both runs. The largest cross-run pLDDT difference was 0.06 on the native 0–100 scale. |
| Terminal structures and fresh final scoring | The six final survivors had identical sequences and numerically coincident Cα coordinates across the two runs. With one rank-zero terminal per replicate, Low, Medium, High and Beam each produced two distinct AF3 passers in two attempts. |
| Fixed small-molecule gallery output | Independent native final scoring produced all four required values and a computational pass. Backbone RMSD was 0.511 Å. |
| 1Y26 AF3 regression | 0.681 Å RMSD against the crystal, within the upstream 0.75 Å limit. Unreasonable bond fraction and clash fraction were both zero. |

The two successful beam main calls took 483 and 511 seconds (0.276 device-hours combined). These wall times include native setup and oracle work and apply only to this small monomer diagnostic. The one-shot smoke times include process startup. They have different timing boundaries from the paper’s warm generation rates. No research leaderboard point is derived from these latencies. Failed infrastructure trials are retained as diagnostics and excluded from biological outcomes.

[Raw numeric receipts](results.json) · [CPU-only receipt checks](../../tests/test_gpu_pilot_evidence.py)

## Replay the model-output comparison

Use the native task's Python dependencies, plus Diffusers commit `537887041938fc6fe61ad583c3ae77477c9ebac9` and the NVIDIA preprocessing package at `199dfd7f83db1487a02fc1f9f15536e32a9a6a6b`. These pins come from the official examples manifest. The model revision is `92fc3ae50e336b2ec59c713c19cfd43e5d3e923b`. Its recorded native source safetensors digest matches the weights in the Harbor task. [Published input and generated CIF](https://huggingface.co/nvidia/RFD4-Proteina-Diffusers/tree/92fc3ae50e336b2ec59c713c19cfd43e5d3e923b/examples/unconditional) · [Examples manifest](https://huggingface.co/nvidia/RFD4-Proteina-prep/blob/c99a272df90566bbcabf4af0e06bb26e58d704b9/examples/manifest.json)

From the task directory, supply a local copy of that release containing the files listed in [fixture.sha256](fixture.sha256). Choose a fresh output directory.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python docs/gpu-pilot/replay_monomer.py \
  /path/to/pinned-diffusers-bundle /path/to/new-run
```

The driver verifies the input CIF, archived output, model files and scheduler configs. It runs the published BF16, seed-42 example, a same-device repeat and a seed-43 control. It aligns by matching backbone atom identity. It compares all atom coordinates when the sequences and atom sets agree. Sampling time uses synchronized GPU calls and excludes loading, input and output I/O. The local repeat check tests equality at the coordinate precision stored in the CIF. The authors report an H200 for the archived example. No acceptance tolerance for cross-device sequence identity is specified for this fixture.

## Evaluation checks

- **Public monomer.** Score the publisher's output, local seed 42, its repeat and local seed 43 with the task's independent AF3 3.0.2 scorer and Foldseek 10-941cd33. Two entries intentionally contain the same local result. This checks duplicate handling.
- **Native beam.** Use the pinned upstream monomer example on its one unconditioned 100-residue input with the native 0.0.3 release. Pin generation and internal AF3 seeds to 42, use eager bf16 and zero data-loader workers, and retain the upstream branching, rewards and save rules. Local process transport runs the upstream persistent worker in the official CUDA 12 oracle runtime on the same GPU as the generator. An instrumented repeat observes the native selection indices. The held-out check designates one terminal output per replicate by the highest checkpoint reward with stable ties. Intermediate rollouts are excluded from this count. [Upstream monomer config](../../environment/references/upstream/inference_beam_example.yaml)
- **Paired one-shot controls.** Run the native Low, Medium and High released thinking aliases separately on the same unconditioned 100-residue input with two replicates, seed 42, batch one and eager bf16. Compare two terminal outputs per method under a fresh common AF3 seed (314159) and cluster within each method. Changing model settings changes the mapping from random seeds to samples. These tiny cohorts test program behavior and have insufficient resolution to fit a scientific performance curve.
- **Native final scoring.** Re-evaluate the immutable NVIDIA small-molecule gallery output under the released AF3 evaluation criterion. The design and input digests are recorded. The upstream evaluator made the AF3 job and computed its final pass decision. This selected example has no identified matching original trial in the Figure 2 target export. Its result measures the scoring stage only.
- **Upstream crystal control.** Run the single-sequence, untemplated 1Y26 case on the deposited crystal using the reference's loading, atom matching, seed 1, RMSD, bond and clash checks. It is a subset of the upstream regression suite. [Upstream assertions](https://github.com/RosettaCommons/RFD4-Proteina/blob/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/tests/e2e/test_af3_structure_quality.py)

The native oracle tests used Google's AF3 parameters, five recycles, one diffusion sample, XLA attention and NVIDIA's versioned CUDA 12 oracle image. Its pinned squashfs SHA-256 is `6013f17d43d5be70fcad6bb9c674a96803023c418ea0c3dde08f8e1d61ad4719` at Hub revision `598acab93d5d0aeacc4362f6e88dc20e7db85ef6`. Local transport used the image's userspace and Python with the host driver in an offline user namespace. The released worker logged a Tokamax autotuning fallback on this host and completed inference with default kernels. The task's standalone Google CLI remains the independent monomer endpoint. [Release runtime provenance](https://github.com/RosettaCommons/RFD4-Proteina/blob/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/scripts/oracle_artifacts.sh) · [AF3 terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)

Initial checks with the upstream batch worker and unmodified Google AF3 Python packages failed at featurization. Version 3.0.2 lacked the worker's glycan keyword and 3.0.3 lacked the OpenFold compatibility keyword even when using AF3 weights. In both cases the native evaluator wrote a zero pass flag with missing required metrics. A final evaluation after a successful prefilter is indeterminate until all required metrics are available. The attempt-level importer supports the two native status fields to catch this failure. A genuine native prefilter rejection still consumes its attempt and cost. The pilot did not change the standalone monomer scorer or reinterpret historical aggregate rows.
