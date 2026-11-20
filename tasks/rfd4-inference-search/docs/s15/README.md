# Table S15 unconditional sampler experiment

This experiment is restricted to three fixed length-only monomer inputs. Following supplementary Table S15, each sampler has 50 original attempts at each of 100, 150 and 250 residues. AF3 backbone scRMSD ≤ 1.5 Å is a computational pass. The primary local metric counts Foldseek clusters containing a passer within each length, divided by all 150 slots. This experiment has a separate compute allowance from the 180-pass-per-slot Harbor task. [RFD4-Proteina supplementary §§S5.1.1 and S12](https://www.biorxiv.org/content/biorxiv/early/2026/11/15/2026.11.12.766037/DC1/embed/media-1.pdf?download=true)

| Sampler | Steps | Numerical specification |
| --- | ---: | --- |
| Stochastic TSR | 60 | Released stochastic sampler with g = β/r². |
| Stochastic TSR, prior schedule | 60 | Same TSR update and time mesh. The schedule returns the original β, with injected variance multiplied by 0.3 for the backbone and 0.1 for the latent. |
| ODE TSR | 60 | Native TSR update with g = 0 for both modalities. |
| Full-length SDE | 400 | Native score-based SDE with unscaled β and the same two variance multipliers. |
| Shortened SDE | 60 | Same native SDE and variance multipliers at 60 steps. |

**Release-specific choices.** We use the public native 0.0.3 checkpoint, released default modality time-grid settings and modality-specific β functions, eager BF16, batch size five and the released medium recycling profile. The 60-step rows have 180 trunk passes per slot. The paper does not give an exact historical checkpoint, seed list or 400-step recycling schedule for Table S15. For 400 steps we proportionally expand the cumulative boundaries of the same profile, giving 1,199 passes per slot. Results measure the stated reconstruction on the public release. The 400-step row cannot be submitted under the live task's standard compute cap. [Release implementation](https://github.com/RosettaCommons/RFD4-Proteina/blob/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/src/rfproteina/flow_matching/product_space_flow_matcher.py)

The local scorer fits N, Cα, C and O across the generated and refolded backbone. Missing or noncanonical native outputs retain their original slots as failures. In the reused 150-sample baseline the generated structures omitted terminal OXT while their AF3 refolds contained it. The four common backbone atoms are measured throughout.

RFD4 uses seed 727000 + length. AF3 uses seed 727000, five recycles, one diffusion sample, empty MSAs and no templates. Foldseek 10-941cd33 clusters all valid generated structures separately per method and length. The score includes every original slot, including failures. The existing stochastic TSR baseline used this configuration. Its saved design/refold atom coordinates and cluster memberships were rechecked. New cohorts time native prediction with CUDA synchronization. Warm sampling times exclude the first batch of each length, process startup, output files and final AF3. End-to-end wall times and authors' reported seconds are distinct measurements.

## Reproduction

Use the task's pinned generation and AF3 environments and materialized data. From the task root run the CPU schedule preflight and the following commands for each method and length. Give each generation or scoring job its own GPU. Use a fresh work root for a new campaign.

```bash
python docs/s15/check_schedules.py
python docs/s15/run_cohort.py ode-tsr 100 generate --assets /workspace/data/assets --root /work/s15
python docs/s15/run_cohort.py ode-tsr 100 score --assets /workspace/data/assets --root /work/s15
python docs/s15/collect.py /work/s15 /work/s15-report
```

The collector uses the checked-in 150-slot stochastic TSR run when that method has no directory in the new campaign. It verifies all per-slot outcomes against the scorer's counts and requires the remaining four complete 150-slot cohorts. Preserve the native config, native manifest, batch timings, AF3 log and cluster assignments for each cohort. AF3-derived observations are subject to the [AF3 output terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md).
