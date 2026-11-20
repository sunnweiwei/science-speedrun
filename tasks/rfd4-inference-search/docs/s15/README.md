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

The local throughput including AF3 divides 150 attempts by the sum of elapsed generation and AF3 GPU-hours across the three lengths. Each length had its own generation and AF3 process, including startup and AF3 compilation. The yield column uses distinct passing clusters in the numerator with the same pooled hours. CPU-only postprocessing is excluded. This S15 monomer rate and the mean of per-target yields in Figure 2 have different aggregation rules.

## Measured operating points

Each row below has 150 fresh generations and 150 final AF3 refolds. The five rows used 6.57 GPU-hours of generation and final AF3 across 15 cohorts. [Full-precision operating points](../../environment/references/s15/summary.csv) · [By length](../../environment/references/s15/by_length.csv) · [All 750 slots](../../environment/references/s15/slots.tsv)

| Sampler | AF3 passes | Passing clusters | Local rate | S15 rate | Attempts/GPU-h including AF3 | Passing clusters/GPU-h including AF3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stochastic TSR | 111 | 73 | 48.7% | 34.7% | 175.2 | 85.3 |
| Stochastic TSR, prior schedule | 0 | 0 | 0.0% | 0.0% | 177.8 | 0.0 |
| ODE TSR | 121 | 70 | 46.7% | 33.3% | 174.5 | 81.5 |
| Full-length SDE | 126 | 74 | 49.3% | 37.3% | 47.5 | 23.4 |
| Shortened SDE | 1 | 1 | 0.7% | 0.0% | 176.8 | 1.2 |

The full-length SDE gained one distinct passer over stochastic TSR in this draw. Stochastic TSR delivered 85.3 distinct passers per GPU-hour and had the highest clustered pass rate of the four 60-step settings. Local warm generation took 10.06 s per design for stochastic TSR and 65.27 s for full-length SDE (6.49×). The paper reports 2.99 s and 17.68 s (5.91×). The three highest local rates have the same order as in S15 and exceed its absolute values by 12.0–14.0 percentage points. [Measured quality–throughput plot](../../environment/references/s15/quality-time.png) · [Pinned assets](../../environment/references/s15/assets.json)

## Repeatability

The earlier public stochastic TSR run produced 82 passing clusters (15, 31 and 36 by length). A coordinate and cluster recheck recovered all 82. The newly generated comparison run produced 73 (15, 28 and 30). Its resolved Hydra settings matched the earlier run apart from file locations, and the input bytes matched. The cause of the difference between campaigns has not been isolated. Both results are retained. [Between-campaign check](../../environment/references/s15/default-repeat.json)

A separate same-device repeat of the new 150-residue cohort regenerated all 50 sequences and their serialized backbone coordinates exactly. A fresh final AF3 run changed individual scRMSDs by up to 0.527 Å. All 50 pass decisions and the entire Foldseek partition agreed, yielding 28 passing clusters both times. This extra cohort is excluded from the five-method comparison. [Same-device control](../../environment/references/s15/same-device-repeat.json)

## Reproduction

Use the task's pinned generation and AF3 environments and materialized data. From the task root run the CPU schedule preflight and the following commands. Give each concurrent cohort its own GPU. Use fresh work and report directories. The AF3 subprocess defaults to `/opt/af3env/bin/python` and `/opt/af3/run_alphafold.py`. Outside the image, set `AF3_PYTHON` and `AF3_SCRIPT` to the corresponding local paths.

```bash
python docs/s15/check_schedules.py
for sampler in stochastic-tsr prior-tsr ode-tsr full-sde short-sde; do
  for length in 100 150 250; do
    python docs/s15/run_cohort.py "$sampler" "$length" generate --assets /workspace/data/assets --root /work/s15 || exit 1
    python docs/s15/run_cohort.py "$sampler" "$length" score --assets /workspace/data/assets --root /work/s15 || exit 1
  done
done
python docs/s15/collect.py /work/s15 /work/s15-report
python docs/s15/plot_results.py /work/s15-report/summary.csv /work/s15-report/quality-time.png
python docs/s15/check_coordinates.py /work/s15 /work/s15-report/coordinate-check.json
```

The collector uses the checked-in 150-slot stochastic TSR run when that method has no directory in the new campaign. It verifies all per-slot outcomes against the scorer's counts and requires the remaining four complete 150-slot cohorts. Preserve the native config, native manifest, batch timings, AF3 log and cluster assignments for each cohort. AF3-derived observations are subject to the [AF3 output terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md).

The optional coordinate check needs raw files for all five local runs. It parses the original generated and AF3 CIFs with Gemmi, recomputes their least-squares backbone fits and cross-checks AF3 query identities and Foldseek memberships against the saved slot observations.
