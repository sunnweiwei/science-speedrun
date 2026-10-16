# Data and fixed input features

The complete prepared data are already mounted read-only under `/workspace/data`. The manifests
store absolute `/data/...` paths; `/data` is a link to `/workspace/data`, so they resolve as they
are. Do not download or materialize data, model weights, MSA databases, or ESMC. Available runtime
layout (under `/workspace/data`):

```text
manifests/foldingtrain-100k-v1/
  foldingtrain_100k.jsonl
  features.structures.jsonl
  features.esmc6b_final_int8.jsonl
objects/openfold3/...                    # 90,000 structure NPZ files
features/esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1/
  <sequence-sha256>.safetensors         # Complete training feature store
foldbench/
  suite/suite_manifest.json
  suite/requests/<request-id>.json
  suite/targets/<track>.csv
  ground_truths/<request-id>.cif
  features.esmc6b_final_int8.jsonl
  features/esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1/
    <sequence-sha256>.safetensors       # Validation-split feature store
models/
  esmfold2-fast-b28d8ace/               # Reference/baseline only
    config.json
    model.safetensors
    README.md
```

There is no `hf/` cache and no ESMC-6B checkpoint. The ESMFold2-Fast directory is
large reference data, not a candidate initialization or deliverable.

## Training pool

- Pool: 100,000 logical records and 90,000 unique structures. Repeated views are
  intentional. All records may be used for training; validation splitting is
  the researcher's choice.
- Manifest records contain `example_id`, source, structure path, entity
  sequences/types, task labels, and selection metadata. NPZs contain OpenFold3
  preprocessed structures; see the pinned reference adapters.
- Protein input features are the frozen ESMC-6B final normalized hidden state,
  stored as model-independent group-64 symmetric INT8. The decoded tensor is
  **`[L,2560]`**. It contains no ESMFold2 projection or layer mixture. The exact
  sequence SHA-256 is the key.
- Every feature row is ready and every referenced tensor is present. Treat any
  missing file as an input error; never load or download ESMC-6B.
- Nonprotein entities use model-designed inputs. No MSA or template database is
  part of the task.

Canonical 100K training-pool manifest SHA-256:
`c2b7e935b130fc5f423916beb2577f7e06afb40943fde7c1d55192604e17a229`.

The mixture and leakage filtering are project-specific, not an official
OpenFold3 or ESMFold2 release. The experimental PDB portion uses a 2021-09-30
release cutoff, and protein homology filtering uses at least 40% identity / 80%
coverage against FoldBench. Known limits include exact-sequence rather than
RNA-family exclusion and incomplete ligand scaffold/pocket-similarity exclusion.
These limitations must accompany benchmark claims.

## Fixed-feature semantics

ESMC-6B was run before the research session. Each safetensors file contains:

```text
values: int8     [protein_sequence_length, 2560]
scales: bfloat16 [protein_sequence_length, 40]
```

Feature identity:
`esmc6b-final-ln-int8-g64-45b0fa5d-ptsdpa-v1`.

Decode each contiguous group of 64 values with its BF16 scale. The unquantized
BF16 master is organizer-only and is not mounted, published as default task
data, or passed to evaluation.

The ESMC-6B checkpoint is not present. The fixed tensors are ordinary input data.
A candidate may use or ignore them but may not regenerate them by adding ESMC to
its package.

The ESMFold2-Fast checkpoint at `/workspace/data/models/esmfold2-fast-b28d8ace` is for
baseline measurement, inspection, and legitimate training-data analysis only.
Candidate weights must not inherit from it, and the submitted inference package
must not contain or call it.

The released checkpoint expects an ESMC layer mixture and its own learned
projection; the task's model-independent final-hidden tensors are not that
input. Therefore, use the supplied ESMFold2-Fast reference results for direct
baseline comparison unless genuinely compatible inputs are available. Never
silently substitute final-hidden features into the released projection and
report the result as the released baseline.

## FoldBench validation split

FoldBench is split into a validation part, mounted here, and a test part that only the verifier
sees. `/workspace/data/foldbench/suite` holds the validation part: 74 coordinate-free assembly
requests and 108 target rows, a few examples of every track (ten for most tracks, two or three for
the DNA and RNA monomers).
`/workspace/data/foldbench/ground_truths` contains their native mmCIFs. They are for development
analysis, evaluation and model selection, not training. The test part (1,448 assemblies, 1,715
target rows) is what the verifier scores; it is not in this workspace.

The split keeps related assemblies on one side: assemblies from the same PDB entry, or with an
identical or highly similar chain, are assigned together, so no validation structure is a near
copy of a test one.

`/workspace/data/foldbench/features.esmc6b_final_int8.jsonl` covers every unique protein
sequence in the validation requests. Its paths resolve within `/workspace/data/foldbench` and use
the same fixed feature identity as training. For replay, relevant rows are staged
in the input-side feature manifest defined by `contract.md`. Other conforming
suites use the same interface with their own precomputed tensors; the candidate
never needs the ESMC checkpoint.

`suite_manifest.json` records the split (its name, the validation fraction and seed). Default run policy is seed 101 and one sample per assembly.

