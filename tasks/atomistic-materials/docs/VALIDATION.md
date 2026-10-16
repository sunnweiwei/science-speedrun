> Inherited PR #90 measurements for **pretrained adaptation**, before the pretraining integration.
> They do not validate scratch convergence or the updated full verifier. See
> [VALIDATION-pretraining.md](VALIDATION-pretraining.md) for current checks. The κ scorer now uses
> reference-owned quadrature weights; historical predictions have not been rescored here. The
> historical mixed-replay development split also used unreduced compositions (see its result note).

# Validation record (v2)

All runs on one host with eight H100 80 GB GPUs, in the task's runtime image (`environment/Dockerfile`:
FAIRChem 2.16.0, phonopy 3.5.1, phono3py 3.30.0, torch-dftd 0.5.1; MP2020 scoring in pymatgen
2023.5.10). Durations are wall-clock on shared GPUs unless stated.

## 1. v1 suites unchanged by the runtime upgrade

phono3py 3.30.0 needs phonopy 3.5.1 (v1 pinned 2.x). The #49 3-epoch checkpoint was re-run on the
v1 validation panel (1,200 cases) and scored with v1's scorer:

| Metric | v1 record | v2 runtime |
| --- | ---: | ---: |
| discovery F1 | 0.8563 | 0.8563 |
| bulk / shear modulus MAE (GPa) | 4.852 / 9.990 | 4.852 / 9.990 |
| phonon max / mean frequency MAE (K) | 53.682 / 19.767 | 53.660 / 19.822 |

## 2. New suites reproduce the study implementations

The suite drivers in `environment/evaluation/extra.py` were written from the scripts used to
measure headroom (`docs/EVALUATION_DESIGN.md`). On the 40 κ validation materials, the per-material
κ differs from the study runs by:
- UMA-S-1.1: median 0.11 %, at most 1.6 %;
- 3-epoch model: median 0.16 %, at most 9 %, on one soft-mode material; GPU relaxations are not
  bit-reproducible.

Censoring agreed on all 40 materials.

MOF bulk modulus, UMA-S-1.1: the study used a different set of 40 MOFs. On the 16 MOFs both runs share,
the MAPE is 20.21 % (study) and 20.14 % (task runner). AM26 validation per-system force RelRMSE for UMA-S-1.1:
a-C 18.6 %, a-Si 11.1 %, a-GST 16.4 %. The study gave 18, 11 and 16 % on all structures.

## 3. Verifier

`tests/` was run as compose would run it: an offline `main` with GPUs and label-free test inputs, an
offline `grader` with references, and a shared sealed volume.

| Run | Result |
| --- | --- |
| smoke: UMA-S-1.1 package, validation data as "test", AM26 + cleavage | sealed and graded; metrics identical to the direct runs (cleavage 4.9600 %, AM26 force RelRMSE 11.9123 %) |
| package without `calculator.py` | `invalid_submission`, all 15 metrics null, reward 0, in seconds |
| `make_calculator` raises | AM26 and NEB scored as all failures (force RelRMSE 200 %, NEB 2.0 eV); physical reported invalid; no hang |
| full test run, mixed-replay package | all eight suites complete. Physical: 6,000 / 6,000 cases succeeded. Other suites: zero failed cases. 6 h 18 min on eight H100s shared with three other evaluation jobs |

Two defects were found and fixed this way:
- a crashed worker left the physical driver blocked on its queue;
- the v1 test-panel files are mode 600, which uid 10001 cannot read. The verifier now copies
  inputs into a readable work tree, and the release tars normalise file modes.

## 4. Baselines

All three baselines were scored on every validation and test suite with the task's drivers. The
mixed-replay test row comes from the full verifier run above. Every case completed; there were no
failed or censored cases outside κ's own symmetry gate. Tables: `environment/baseline/*/RESULTS.md`
and README §6.

The mixed-replay baseline was trained in this setup: `prepare_mixed.py` (16 min), then FAIRChem on
eight H100s (2 h 17 min; 46,875 updates; development loss 0.988 at step 5,000 → 0.481 at the final
step, which was selected).

On the validation split, the 3-epoch model is better than UMA-S-1.1 on NEB (0.147 vs 0.191 eV);
on test they are equal (0.126 eV). The study's 163-path comparison had suggested the opposite
direction, so NEB headroom is small, as the design notes said.

## 5. Release

`materialize.sh release` path: a full install of the published release from Hugging Face
(`release.py install`, every artifact checked by size and SHA-256, then `validate`) finished in
40 min, giving 89 GB and validation PASS (42,409,826 pool frames). The pool shards are byte-identical
between the first and the pinned revision; only the five small tars were re-published with readable
file modes. A partial install of val, test, models and baselines is byte-identical to the release
directory.

## 6. Data checks

- Pool: every source compared with UMA-S-1.1 predictions on sample frames before release:
  - MatPES stress sign flipped;
  - MPLiTrj dropped (force correlation 0.39);
  - sAlex pressure correlation 0.92 and force MAE 0.02 eV/Å.
- Exclusions: `pool/blocked.json` has 178,802 compositions and 278,213 material IDs.
  MatPES kept 193,171 of 434,712 frames (it samples MP materials); the original 500k MP sample lost
  1,159 + 5 frames.
- MOF CIFs: GitHub served them gzip-encoded. The builder now stores decoded files; all 100 parse.
