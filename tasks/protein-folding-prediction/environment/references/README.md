# References: approximately-50M architectures and prior experiments

The folding architectures, pinned upstream sources and prior controlled experiments below have not
been re-run by the integrator. Treat their numbers as context, not as checked results. The
verified starter and the frozen ESMFold2-Fast reference numbers are in
`/workspace/baseline/starter/RESULTS.md`. Read the executable starter first, then use these
references to decide what to reproduce, retain, or replace.

## Recommended executable starting point

`/workspace/baseline/starter` is a from-scratch, no-MSA adaptation of Biohub's released
ESMFold2 module graph. It keeps the upstream all-atom input embedder, pair trunk,
PARCAE/CODA recurrence, diffusion structure module, distogram head, and
confidence modules at approximately 49M folding parameters. Its language-model
adapter consumes the canonical evaluator-supplied ESMC-6B final hidden state
instead of loading ESMC or copying ESMFold2's released layer-mixture projection.

The starter includes:

- exact parameter auditing;
- current FoldingTrain-100K native-view construction;
- eight-GPU DDP training with BF16 activations, explicit token/atom memory
  guards, and EDM coordinate plus distogram losses;
- ordinary and EMA checkpoint/resume state;
- one-current-candidate publication; and
- offline all-atom replay through `submission/infer.sh`.

Run its documented audit and smoke tests before changing the architecture.

## Pinned upstream source trees

The development image contains writable, network-independent source snapshots:

| Path | Architecture | Pinned revision |
| --- | --- | --- |
| `/opt/transformers` | Biohub ESMFold2 module graph | `ef32577f55da19a4989cd7b22e004dc43a4998cb` |
| `/opt/esm` | Biohub biomolecular input/decoding stack | `917af90b624535eed1e072d343c717e3ec11fef4` |
| `/opt/references/Protenix` | Pairformer + atom diffusion | `c3bfc365b3e1341a11935eddfe7bfdc308092147` |
| `/opt/references/OpenDDE` | Pairformer + structural-token diffusion | `f607bb3c9ff299c0627ac20f5ef8e25d716ed46f` |
| `/opt/references/dplm` | DPLM/DPLM-2 discrete structure modeling | `8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d` |

The ESMFold2 and ESM sources are installed and used by the executable starter.
The other upstream trees are supplied for architecture and native-training-code
study. Copy any code required by a final candidate into its package or use only
dependencies guaranteed by the task image.

## Previously validated integrations and results

`/workspace/references/prior_workspace` is a source snapshot of the earlier
validated 50M workspace at clean source paths from
`sunnweiwei/protein-folding` revision
`1e21e527914a49b02acb4b79bcbdada805944cd5`. It contains:

- `official_baselines/esmfold2`: 49.2M Biohub ESMFold2 graph;
- `official_baselines/protenix`: 48.7M Protenix/Pairformer diffusion;
- `official_baselines/opendde`: 48.7M OpenDDE structural-token diffusion;
- `official_baselines/dplm2`: 47.1M DPLM-2 deployment core;
- `first_party_baselines/hybrid_esmc_pairformer`: 48.8M ESMC + Pairformer;
- controlled `README.md` and `RESULTS.md` records; and
- machine-readable architecture specifications under `model_specs/`.

Those files preserve the architecture maps, training wrappers, results, and
provenance that informed the earlier experiments. Some old data adapters refer
to the retired mixed-256 feature cache and old ModelBundle contract. Do not run
them blindly. The executable starter is the maintained adapter for this task's
model-independent final-hidden INT8 features and replay contract.

## Scientific boundary

Reference source and configurations may be copied or changed. Reference learned
weights may not initialize, be transformed into, or be included in a candidate.
The supplied ESMFold2-Fast checkpoint under `/workspace/data/models` is therefore a
comparison/analysis asset, not the starter initialization. All starter and
candidate weights begin from random initialization in the current session.

The 45M--55M range is a task requirement, not a claim that parameter count alone
ensures quality. The intended research space includes representation, pair and
atom reasoning, structure generation, multimodal featurization, objectives,
data construction, optimization, inference-time compute, and checkpoint
selection.
