# Model replay contract

## Continuously published candidate

Continuously maintain the strongest complete candidate under
`/workspace/submission/`:

```text
submission/
  infer.sh                 # Required batch inference entry point
  README.md                # Architecture, size, training provenance, limitations
  setup.sh                 # Optional dependency installation; no model downloads
  ...                      # Candidate code, config, weights and owned assets
```

Keep this directory replayable at all times while research continues. Evaluation
copies an instantaneous version into a fresh environment and replays it
independently. Do not depend on the live training process, optimizer state, files
outside the package, an unrecorded cache, or a machine-specific absolute path.
Resolve model-owned files relative to the submission directory. Symlinks must
not escape it.

The package must contain exactly one candidate model's inference-time learned
parameters. Those parameters must have been trained from scratch during this
session; they may resume from earlier checkpoints of that same candidate lineage.
The package must not contain or load ESMC-6B, ESMFold2, ESMFold2-Fast, or any
other externally pretrained parameters, whether copied directly or transformed.
The fixed ESMC final-hidden INT8 input tensors described below are evaluator-supplied
data, not model parameters, and must not be copied into the package.

The candidate must contain 45,000,000--55,000,000 learned parameters. Count every
learned parameter needed for inference, including a projection from the supplied
ESMC features, structure and confidence modules, and all modality-specific
components. The frozen evaluator-supplied feature tensors are data and are not
parameters. The package must report the exact count in `README.md` and provide a
reproducible parameter-audit command.

This directory is the current candidate, not a checkpoint archive. It must not
contain historical `candidate-vN` directories, superseded weight files,
optimizer states, or rollback packages. Store those outside `submission/`;
Runner snapshots preserve earlier published states.

The optional `setup.sh` may install pinned software dependencies. It must not
download model weights or require a secret. After setup, inference runs without
network access. The candidate architecture and checkpoint format are otherwise
unrestricted. The starter Python API is optional; `infer.sh` is the contract.

## Invocation

```bash
bash /path/to/submission/infer.sh /path/to/input /path/to/output
```

Both arguments are absolute directories. Treat input as read-only; output is an
empty writable directory. The caller's current directory is unspecified. Exit
zero on a completed batch and nonzero on an unrecoverable error. Do not modify
model parameters during inference. Use the same fixed parameters for every
request and track.

## Resources

The verifier calls `infer.sh` once for the whole suite: about 1,500 requests, the largest about
7,000 tokens after chain copies and ligands. It runs on eight H100 80 GB GPUs with 256 GB of host
memory and no network. The batch must finish within **8 hours**. A run that exceeds the time, or is
killed for exceeding the memory, is an invalid submission and scores 0. Size inference so that every
request fits these limits, for example by chunking the pair representation or by recording a
request the model cannot serve as a missing prediction (a failure for that target, not for the
batch).

## Inputs

`input/requests.jsonl` contains one JSON object per line. Each object follows
`foldbench-folding-request/v0`, inherited from the pinned source project's
`folding_contract.py`:

```json
{"schema_version":"foldbench-folding-request/v0","request_id":"unseen-example","target":{"entities":[{"id":"E1","type":"protein","sequence":"ACDEFG","copies":1}]},"run":{"seeds":[101],"num_samples":1}}
```

- `request_id`: unique filesystem-safe ID matching `[A-Za-z0-9][A-Za-z0-9_.-]*`.
- `target.entities`: nonempty array of protein/RNA/DNA sequences or ligands.
  Entities have unique `id`, `type`, and optional positive `copies` (default 1).
  Polymers have `sequence` and optional `modifications` entries with 1-based
  `position` and `ccd`. Ligands have exactly one of `ccd_codes` or `smiles`.
- Optional `target.covalent_bonds` joins `left`/`right` atom references, each
  with `entity_id`, 1-based `position`, and `atom_name`.
  When the referenced entity has multiple copies, an endpoint may add a
  1-based `copy` (default 1) to select the expanded copy.
- `run.seeds`: unique nonnegative integers; `run.num_samples`: positive integer.
  Produce samples indexed `0 .. num_samples-1` for each supplied seed.

Every unique protein sequence in the batch has one evaluator-supplied frozen
ESMC-6B final-layer feature tensor. `input/features.esmc6b_final_int8.jsonl`
contains one record per sequence:

```json
{"sequence_sha256":"443c1c9d...","length":6,"shape":[6,2560],"dtype":"int8","scale_shape":[6,40],"scale_dtype":"bfloat16","quantization":"symmetric-int8","group_size":64,"feature_version":"esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1","path":"features/esmc6b-final-int8/443c1c9d....safetensors"}
```

`path` is relative to the input directory and names a nonempty safetensors file
with `values: int8 [L,2560]` and `scales: bfloat16 [L,40]`. Decode group 64 as
`values.float().reshape(L,40,64) * scales.float().unsqueeze(-1)`, then reshape
to `[L,2560]`. This is the ESMC-6B model's final normalized hidden state; it
contains no ESMFold2 projection or layer mixture. Match a
protein to its record with `sha256(sequence.encode("utf-8")).hexdigest()`.
RNA, DNA, and ligand entities have no ESMC tensor. A model may use or ignore
these features, but it cannot require the ESMC checkpoint.

The same input format applies to visible and private suites. Request IDs,
sequences, feature files, lengths, ordering, batch size, and seeds may all differ
from the development suite. The evaluator does not pass target tables, native
coordinates, benchmark answer tables, or cached predictions as inference inputs.

## Outputs

Write `output/prediction_reference.csv` with exactly these columns:

```csv
pdb_id,seed,sample,ranking_score,prediction_path
unseen-example,101,0,0.7,structures/unseen-example-101-0.cif
```

`pdb_id` is the request ID, not necessarily a real PDB accession. Each
`(pdb_id, seed, sample)` must be unique and requested. `ranking_score` is finite;
larger is better and must not use ground truth. Paths are relative to the output
directory, remain inside it after resolving symlinks, and name nonempty mmCIFs.

Structures must represent the requested assembly, including polymer copies,
ligands, appropriate atom/residue/chain identities and finite all-atom coordinates.
Provide valid mmCIF entity/polymer metadata (`_entity`, `_struct_asym`,
`_entity_poly`, `_entity_poly_seq`, as applicable), not just an `atom_site` table.
The official OpenStructure-based scorer must be able to parse the structure.
CA-only scaffolds are not complete biomolecular predictions.

The verifier scores each of the nine tracks with the pinned official FoldBench scorer (lDDT for
monomers, DockQ or ligand success rates for interfaces, each in [0, 1]). A protein or nucleic-acid
interface succeeds when DockQ ≥ 0.23 and its geometry is CAPRI acceptable: ligand RMSD ≤ 10 Å or
interface RMSD ≤ 4 Å. Chains collapsed onto each other recover every native contact and pass
DockQ alone; they do not pass this rule. The reward is the
unweighted mean of the nine track scores. Each track score is reported next to it, and the research
objective is to improve every track, not only the mean.

Missing or invalid predictions count as failures, not a smaller benchmark. The
scorer retains every original target row in its denominator and uses model-
provided ranking, never oracle best-of-ground-truth selection.

## Replay and review boundary

The evaluator freezes one instantaneous copy of `/workspace/submission` and uses
it for all tracks. Contract validation checks packaging and replayability;
from-scratch training provenance, absence of transformed pretrained parameters,
benchmark-data compliance, and the one-model rule also require code/checkpoint
review and cannot be proven solely by successful execution.
