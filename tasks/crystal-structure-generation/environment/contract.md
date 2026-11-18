# Frozen crystal-generation contract

Maintain a complete package at `/workspace/submission` containing:

```text
generate.sh
model_manifest.json
README.md
model code, one checkpoint, configuration, and owned offline assets
```

The package may be read at any moment. Replace it as a whole: assemble it elsewhere on the same
filesystem and swap it in with one rename, as the starter's `publish.py` and `train.py` do.
A package that fails validation, exits nonzero or exceeds the inference limits scores 0.

The verifier relocates the package and invokes:

```bash
bash generate.sh INPUT_ROOT OUTPUT_ROOT
```

The working directory is unspecified. Input is read-only, output starts empty,
and inference has no network. The package may not call the evaluator, train,
modify its weights, or require files from the development workspace.

Inference limits: the package runs as an unprivileged user from a read-only copy, with no
network, the verifier image's pinned runtime (nothing is installed), up to eight visible GPUs
(`CUDA_VISIBLE_DEVICES`), and 24 hours of wall time for the whole request. Output is limited to
20 GiB and 20,100 files. Processes left behind after `generate.sh` exits are killed.

The grader first relaxes a separate copy of each submitted crystal with the frozen UMA
`uma-s-1` (`omat`), BFGSLineSearch and FrechetCellFilter, up to 200 steps at 0.02 eV/Å and
300 seconds per crystal. It retains finite unconverged iterates and uses the original
candidate when refinement fails. It then scores the resulting geometry with ORB-v3, MACE-MP
and UMA. Mean energy above hull defines stable (≤0 eV/atom) and metastable (0–0.1 eV/atom)
bins, with stable structures excluded from the metastable bin. Let `a = SUN / 2,500` (`ΔE_hull ≤ 0 eV/atom`)
and `b = (SUN + MetaSUN) / 2,500` (`ΔE_hull ≤ 0.1 eV/atom`), where SUN and MetaSUN counts are unique within
their bin and novel relative to LeMat-Bulk over all 2,500 requested slots. The primary score written to
`reward.json["score"]` and `reward.json["reward"]` is:

```text
score = reward = (a + b) / 2 = (SUN + 0.5 * MetaSUN) / 2,500
```

Unweighted `unconditional_sun_metasun` (`b`), `unconditional_sun` (`a`), `unconditional_metasun`,
`negative_formation_energy_fraction` (`E_form < 0`), `relaxation_rmsd`, `as_submitted` pre-relaxation counts,
and element-arity (`n_elements`) distributions are retained as diagnostics in `report.json` (no ad-hoc gates).

For equal compositions, CrystalGen applies MatterGen's v1.0.3 ordered-disordered matcher.
The additional dilute-parent guard identifies a known lattice decorated with a substituent
occupying ≤25% of the host-plus-guest sites. It requires relative radius difference ≤0.3 and
electronegativity difference ≤1.0. Supercells, symmetry copies and dilute variants of a known
compound such as GdDy7Cu8 from DyCu are covered by these comparisons. A cross-element substitution requires both the measured radii and the electronegativities. Ordinary same-species geometric matches still apply to materials such as XeF₂ when a radius is unavailable.

The optional `--as-submitted` evaluation scores the raw lattice and coordinates. It also
measures displacement from the raw geometry under all three potentials on 250 prespecified
slots (50 steps, 0.02 eV/Å), with coverage and a 95% structure-bootstrap interval.
This result is recorded separately from the primary UMA-200 reward.

## Requests

`INPUT_ROOT/requests.jsonl` contains one `crystalgen-request/v0` object per line.

Unconditional example:

```json
{"schema_version":"crystalgen-request/v0","request_id":"unconditional-final","mode":"unconditional","condition":null,"n_samples":2500,"seed":1847}
```

Rules:

- `request_id` is unique and matches `[A-Za-z0-9][A-Za-z0-9_.-]*`.
- `mode` must be `unconditional` and `condition` must be `null`.
- Exactly the six fields shown above are accepted; conditional requests are rejected.
- `n_samples` is an integer from 1 through 10,000 and `seed` is a nonnegative
  integer; booleans are not integers for either field.

Final evaluation uses one 2,500-sample request with a fresh operator-held seed.
It does not depend on held-out structures or target properties. The v0 request
envelope is retained for compatibility with unconditional inputs; the final
suite manifest is `crystalgen-final-suite/v1` and rejects legacy mixed suites.

## Outputs

Write `OUTPUT_ROOT/manifest.csv` with exactly these columns:

```csv
request_id,sample_index,structure_path
unconditional-final,0,structures/unconditional-final-0000.cif
```

There must be exactly one row for every requested `(request_id, sample_index)`
pair and no other rows. `sample_index` spans `0 .. n_samples-1`. Paths are
relative, remain inside the output directory after resolution, are unique, and
point to nonempty regular `.cif` files. Symlinks are not accepted.

The optional `OUTPUT_ROOT/report.json` is diagnostic only and is never trusted
for scoring.

Missing, duplicate, extra, malformed, escaping, or unparsable outputs count as
failed requested samples. Contract-level ambiguity or evidence of tampering may
instead fail verification.

## Model manifest

`model_manifest.json` follows `crystalgen-model/v0`:

```json
{
  "schema": "crystalgen-model/v0",
  "checkpoint": {"path": "model.safetensors", "sha256": "...", "format": "safetensors"},
  "parameter_count": 12400000,
  "one_parameter_set": true,
  "trained_from_scratch": true,
  "training_data": {"name": "MP-20", "split": "train", "source_sha256": "133bab58..."},
  "model_config": {"architecture": "..."}
}
```

Competitive submissions use one `.safetensors` checkpoint. The verifier parses
its tensor shapes and requires their total element count to equal
`parameter_count`, which must be from 1 through 70,000,000. JSON checkpoint
format is reserved for the task's tiny interface smoke baseline. Undeclared
`.bin`, `.ckpt`, `.npy`, `.npz`, `.pt`, `.pth`, or `.safetensors` files are
rejected, as are packages above 2 GiB or 10,000 files.

The checkpoint path must remain inside the package and its SHA-256 digest must
match. Training provenance and disguised transformed-weight reuse still require
code/checkpoint review; those properties cannot be proven from bytes alone.
