# DPLM-2 official-code ~50M baseline

The completed frozen-100K pass-1 training and FoldBench result is recorded in
[`RESULTS.md`](RESULTS.md).

This baseline executes ByteDance's public DPLM/DPLM-2 source at revision
`8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d`.  Local code only materializes a
smaller ESM configuration and delegates to the upstream training and generation
runners.  The multimodal core uses 12 layers, hidden size 512, FFN size 2048,
8 heads, and the official 8,229-token vocabulary.  The frozen official
structure tokenizer is external to the folding-core parameter budget.

The locked non-data recipe follows upstream `dplm2_150m.yaml`: 25% each
single-modality, folding, inverse-folding, and joint examples; 500 diffusion
steps; LoRA rank 16 and
dropout 0.1; `StructAARDMCrossEntropyLoss`; AdamW at 1e-4 with betas
(0.9, 0.98), weight decay 0.01; polynomial schedule with 2,000 warmup and
100,000 total steps; and gradient clipping 0.5.

Training data is deliberately not inherited from DPLM-2. The adapter consumes
the project-wide frozen FoldingTrain manifest and split. Because public DPLM-2
accepts only protein monomers, its audited native view contains only the
supported protein-monomer records from that same split. RNA, DNA, ligand and
complex records remain in the global dataset but are reported as unsupported
for this backend; no PDB+SwissProt examples are substituted.

Strict initialization requires a matching 50M DPLM sequence checkpoint.  A
150M public checkpoint cannot be loaded into the scaled topology.  Therefore a
full reproduction includes the official DPLM pretraining stage at the same
scaled dimensions before DPLM-2 training; random initialization or partial
150M loading is rejected by the wrapper.

That initialization is produced by `pretrain_sequence.py`: MLM stage 1,
followed by DPLM random-mask diffusion stage 2.  Objectives and optimizer
families remain upstream-owned, but the controlled 100K campaign replaces the
paper-scale 500k/100k step counts with `--passes 1`.  Otherwise DPLM-2 would
receive far more data exposure than the other candidates. `export_dplm.py`
converts the resulting upstream Lightning checkpoint to the local Hugging Face
directory format that the unmodified DPLM-2 loader accepts.  The export
contains no parameter transformation.

The released MLM writer and DPLM stage-2 loader disagree on one state
namespace (`model.*` versus the loader's assumed `model.net.*`).
`prepare_mlm_initialization.py` creates an audited compatibility checkpoint by
adding only that namespace and verifies every tensor value is identical.

The same release YAML also contains stale `model.vocab_size` and
`model.freeze_backbone` keys that its strict `DPLM2Config` no longer declares
or consumes. The folding wrapper deletes only these unused keys after guarding
the pinned dataclass shape; the live model derives vocabulary size from the
frozen combined tokenizer, and the released LoRA setup controls trainability.

The public folding route accepts protein amino-acid sequences and emits
discrete protein structure tokens which the frozen tokenizer decodes to
coordinates.  It is not an all-biomolecule predictor, so it must be marked
unsupported—rather than assigned fabricated scores—on FoldBench RNA, DNA, and
ligand tracks.

The finalized common-data view is
`/data/derived/dplm2_50m/folding_view_v1`: 42,912 train and 444 validation
protein monomers. `validate_unified_view.py` loads it through the upstream
`TokenizedProteinDataset` and `DPLM2Collater` and records the resulting batch
schema in the model audit directory.

Build and audit entirely in Docker:

```bash
docker build -f Dockerfile.dplm2 -t protein-folding:dplm2-official .

docker run --rm --network none \
  -e PYTHONPATH=/workspace/official_baselines/dplm2/runtime_stubs:/workspace:/opt/dplm/vendor/openfold \
  -v /mnt/localssd/${USER}/workspaces/protein-folding:/workspace:ro \
  -v /mnt/localssd/${USER}/protein-folding-data/models/dplm2_50m/configs:/configs:ro \
  protein-folding:dplm2-official \
  python -m official_baselines.dplm2.verify --config /configs/dplm2_50m
```

Structure tokenization additionally mounts the frozen IF1 cache read-only at
`/root/.cache/torch` and includes the upstream vendored OpenFold package on
`PYTHONPATH`:

```bash
-e PYTHONPATH=/workspace/official_baselines/dplm2/runtime_stubs:/workspace:/opt/dplm/vendor/openfold \
-v "$DATA_ROOT/source-cache/torch:/root/.cache/torch:ro"
```

`train.py --profile full --passes 1` executes the controlled data-exposure
budget. The `smoke` profile intentionally shortens the run and is never
quality-comparable.
`smoke_step.py` is an even smaller loss-graph diagnostic: it uses the upstream
multimodal model and criterion on synthetic structure tokens and is not a
substitute for an upstream Trainer checkpoint smoke.

After a completed folding run, `export_dplm2.py` restores the Lightning
checkpoint through the released model class, merges the released LoRA adapter,
and writes the deployable Hugging Face weights plus a digest/provenance audit.
The merged deployment parameter count, rather than PEFT's temporary duplicate
`modules_to_save` tensors, is the folding-model budget value.

`contract_inference.py` is the portable one-request ModelBundle entrypoint.
`foldbench_inference.py` is its throughput-equivalent FoldBench adapter: it
loads a bundle once, resolves protein-monomer sequences exclusively from the
coordinate-free FoldBench manifest, deterministically shards the complete
target table, and emits standard `prediction_reference.shard-*.csv` files.
Unsupported FoldBench tracks remain N/S.
`finalize_foldbench.py` fail-closes on missing/duplicate shards and prediction
artifacts, materializes the evaluator's exact supported target subset, and
records both the official denominator and capability rejection coverage before
the pinned FoldBench scorer is invoked.
