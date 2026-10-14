# Independent verification

`test.sh` is Harbor's verifier entry point, not an agent runner. In a fresh
verifier environment it invokes the delivered `infer.sh`, validates predictions,
runs the pinned official FoldBench scorer, and writes `/logs/verifier/reward.json`:
`reward`, the unweighted mean of the scored tracks, next to each per-track score. Detailed diagnostics go to `report.json` and
the `evaluation/` directory. Infrastructure/contract errors exit nonzero and
write `error.json`; they do not masquerade as a successful evaluation.

Harbor transfers `/workspace/submission` from the agent environment. The verifier
does not receive the live workspace, training data, baseline checkpoint, or the
agent's installed packages. The Compose overlay mounts only the benchmark
subtree, including its fixed input-feature tensors. Software dependencies must
be part of the submission or installed by `setup.sh`; that script cannot fetch
model weights.

The evaluator constructs fresh input-only requests and a fixed-feature manifest
whose paths are relative to that input directory. Output paths must remain within
the output directory; arbitrary IDs/seeds are supported. Native coordinates are
not inference arguments. It also rejects byte-identical copies of the supplied
ESMFold2-Fast and ESMC checkpoints; deeper provenance remains review-based.
However, this minimal verifier runs
inference and scoring sequentially in one container, so it is **not a security
boundary against a malicious model reading files or tampering with grading**.
It is intended for this benchmark-visible research task. A future private task
must isolate model execution from labels and the trusted scorer at runtime.

## Scores

Reward JSON has one key per available track, each on a 0–1 scale. There is no
additional overall scalar. For canonical evaluation all nine keys are present.

| Tracks | Score |
| --- | --- |
| `monomer_protein`, `monomer_rna`, `monomer_dna` | Mean lDDT, missing/nonfinite values zero |
| `interface_protein_protein`, `interface_antibody_antigen`, `interface_protein_peptide`, `interface_protein_rna`, `interface_protein_dna` | Fraction with DockQ >= 0.23 and CAPRI acceptable geometry (LRMSD <= 10 Å or iRMSD <= 4 Å) |
| `interface_protein_ligand` | Fraction with RMSD < 2 Å and lDDT-PLI > 0.8 |

The official scorer emits raw rows. We join those rows back to every original
target; missing, invalid and nonfinite predictions cannot shrink denominators.
When multiple samples exist, selection uses the model's ranking score, never
the ground-truth metric. NA interfaces use the official DockQ-v2 output.

## Local interface checks

From the repository root, in an environment with Python and Biopython:

```bash
python -m unittest discover -s foldbench/tests -p 'test_*.py' -v
```

These test the format and adapters with tiny synthetic fixtures. They neither
download the dataset nor train a model or establish a FoldBench score. A full
run needs the pinned HF data release (see `../environment/data/README.md`)
and an actual trained submission. Archive/materialization tests are offline;
they do not run a new training experiment or claim a full benchmark score.
