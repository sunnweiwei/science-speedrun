# Scientific Superintelligence Index v2

The benchmark aggregates twenty independent scientific tasks with equal weight. Each task contributes one overall score. Store a task's complete trusted verifier output under `RECEIPTS/<task slug>/`. The operator-side index reader uses only these receipts and records their SHA-256 digests. It is not part of any Agent container.

```bash
python3 harness/ssi_index_v2.py RECEIPTS --out index.json
python3 harness/ssi_index_v2.py RECEIPTS --partial --out progress.json
```

A missing evaluation is pending. Invalid submissions receive the outcome produced by their verifier. Infrastructure failures do not have a scientific score. The partial view withholds the 20-task mean until a verified task score exists for every task.

## Changed protocols

- **EEG decoding.** Use the four-track chance-adjusted aggregate from a full run's `report.json`. The five native accuracy values remain in `reward.json`. FACED clips recur across subjects and the track is retained as a separate diagnostic.
- **Fluid control.** The standard 26-case suite adds a scalar reward and a signed unclipped task mean. Improvements are relative to the paired zero-action measurement in the correct physical direction. Individual improvements are clipped to [−1, 1] and failures contribute −1. Floor the final mean at zero.
- **Neuron reconstruction.** The native verifier rewards SHAPE's local sparse nERL, as required by both original prompts. The index applies the existing fourth-root display to agents, starter, reference and grid equally. Report physical ERL, errors, coverage and the image-free grid beside the score. A grid outscored all evaluated models on this one core. [ERL in neuron reconstruction](https://doi.org/10.1038/s41592-018-0049-4)
- **Tokamak control.** New evaluations use the maintained v4 native suite and latency check. Historical comparisons explicitly apply its fixed-denominator rule to the 35 scheduled v3 episodes. Differences in devices, perturbations and number of seeds limit comparisons across campaigns.
- **Virtual Cell.** Report calibrated WMSE skill per cell context and average the three. A cross-fitted experimental repeat defines the positive anchor and an across-target mean response defines the negative anchor. These measurements target the mean transcriptional response. The v1 distribution-related panel remains available for interpretation. [Calibration study](https://doi.org/10.1038/s41587-026-03307-w)
- **Mass spectrometry.** Use raw per-spectrum Hit@1 under independent invocations containing at most one spectrum per molecular candidate pool. The frozen test admits 373 batches. The legacy normalized score remains diagnostic.
- **Crystal generation.** Apply the archived UMA-200 relaxation to the model output, use the MACE checkpoint paired with the published energy hull, and apply MatterGen disorder matching plus the SSI dilute-substitution guard. A separate run grades submitted geometries before relaxation. The denominator includes every requested generation attempt. Report SUN, MetaSUN, validity and relaxation displacement separately. [LeMat checkpoint correction](https://github.com/LeMaterial/lemat-genbench/blob/fbf1ba4855934acb8fe87135315504899064080d/README.md)

## Index choices

The task-native protein-design verifier reports four success rates, and the index takes their equal mean for continuity. The native report must identify target-conditioned PPI inverse folding (`proteinmpnn-fixed-target/v2`). Both chains enter ProteinMPNN, only the binder is redesigned, and every generated binder is refolded with the fixed partner in RF3. Report the conditional tracks individually when interpreting a lead. Molecular pretraining uses 1/(1+relative error) with the four group MA-RAEs weighted by 6, 4, 9 and 5 endpoints. Other retained protocols use their native overall values.

The archived PDE truth remains frozen. The independent numerical-solver plus persistence comparison spans all four groups and is distinct from the task starter, trivial predictor and external learned-model portfolio. [Control and reproduction procedure](../pde-surrogate/docs/validation/numerical_controls/README.md)

Score extraction is separate from scientific certification. Include model/source identity, release and data cutoffs, full versus restricted test coverage, estimates and post-hoc specialist routing in published comparisons. The task-specific materializers specify the data needed for a new run.

Each task has a [source-to-measurement entry](SSI_MEASUREMENT_SOURCES.json) identifying the literature measure, the SSI aggregation and, for changed protocols, what the archived Agents were evaluated on.
