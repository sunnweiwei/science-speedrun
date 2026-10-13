# Sycamore surface-code data

The Google Quantum AI Sycamore surface-code memory experiments
([Zenodo 6804040](https://zenodo.org/records/6804040), CC-BY-4.0):

- four distance-3 patches and one distance-5 patch;
- both X and Z bases;
- 13 odd durations from 1 to 25 rounds;
- 130 conditions in total.

Per condition, training takes the first 19,880 even rows and validation the last 5,120 even
rows. The test set is all 25,000 odd rows, and it exists only in the verifier. Arrays do not
expose source row indices.

`/workspace/data` holds this file and the read-only development release:

```text
/workspace/data/
  release.json, inventory.json          release metadata and the SHA-256 of every file
  train/conditions/<condition>/         2,584,400 training shots in total
    arrays.npz                          sample_ids, packed detection events, logical labels
    circuit_ideal.stim                  the ideal circuit
    pij_from_even_for_odd.dem           detector error model fitted on even rows (train + validation)
  train/manifest.json
  validation/conditions/<condition>/    665,600 validation shots, same layout
  validation/manifest.json
  reference/results/                    the reference AlphaQubit's validation predictions, validation
                                        metrics, aggregate test result and training record (no weights)
  reference/package-template/           its submission layout, without weights
```

Condition names read `surface_code_b{X,Z}_d{3,5}_r{01..25}_center_{i}_{j}`. The even-fitted
detector error models use all even syndromes, including validation syndromes, but no test
syndromes or logical labels. They are for streaming Stim simulation. The input format the
verifier supplies at inference (`surface-code-decoding-input/v1`) is specified in
`/workspace/contract.md`. `python -m qecbench.stage_validation` builds that format from
`validation/` (see `/workspace/evaluation/README.md`).

The historical test set is public upstream, so this is an execution boundary, not a secret test.
Do not search the public upstream dataset for test answers.
