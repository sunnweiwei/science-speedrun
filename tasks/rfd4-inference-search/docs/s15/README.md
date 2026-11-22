# Table S15 unconditional sampler experiment

This experiment is restricted to three fixed length-only monomer inputs. Following supplementary Table S15, each sampler has 50 original attempts at each of 100, 150 and 250 residues. AF3 backbone scRMSD ≤ 1.5 Å is a computational pass. The primary local metric counts Foldseek clusters containing a passer within each length, divided by all 150 slots. This experiment has a separate compute allowance from the 180-pass-per-slot Harbor task. [RFD4-Proteina supplementary §§S5.1.1 and S12](https://www.biorxiv.org/content/biorxiv/early/2026/11/15/2026.11.12.766037/DC1/embed/media-1.pdf?download=true)

| Sampler | Steps | Numerical specification |
| --- | ---: | --- |
| Stochastic TSR | 60 | Released stochastic sampler with g = β/r². |
| Stochastic TSR, prior schedule | 60 | Same TSR update and time mesh. The schedule returns the original β, with injected variance multiplied by 0.3 for the backbone and 0.1 for the latent. |
| ODE TSR | 60 | Native TSR update with g = 0 for both modalities. |
| Full-length SDE | 400 | Native score-based SDE with unscaled β and the same two variance multipliers. |
| Shortened SDE | 60 | Same native SDE and variance multipliers at 60 steps. |

**Release-specific choices.** We use the public native 0.0.3 checkpoint, released default modality time-grid settings and modality-specific β functions, eager BF16, and batch size five. In the initial local campaign (`summary.csv`), `run_cohort.py` inherited the user-facing `thinking_medium` recycling profile from `configs/inference.yaml` (180 trunk passes for 60-step rows and 1,199 passes proportionally expanded for 400 steps). Because supplementary §S5.1.1 precedes the recycling study in §S7.2 and the underlying `configs/generation/inference_base.yaml` specifies `n_recycle: 0` (`n_recycle_schedule: null`, i.e., 1 trunk pass per step = 60 passes for 60-step rows and 400 passes for 400-step SDE), we also ran a complete second 750-slot campaign under `n_recycle: 0` (`base_schedule_summary.csv`). [Release implementation](https://github.com/RosettaCommons/RFD4-Proteina/blob/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/src/rfproteina/flow_matching/product_space_flow_matcher.py)

The local scorer fits N, Cα, C and O across the generated and refolded backbone. Missing or noncanonical native outputs retain their original slots as failures. RFD4 uses seed `727000 + length`. AF3 uses seed `727000`, five recycles, one diffusion sample, empty MSAs and no templates. Foldseek `10-941cd33` clusters all valid generated structures separately per method and length. Warm sampling times exclude the first batch of each length, process startup, output files and final AF3.

## Measured operating points: `n_recycle=0` vs `thinking_medium`

Both local campaigns below consist of 750 fresh generations and 750 final AF3 refolds verified with independent Gemmi CIF alignments (`max_abs_scRMSD_difference_A < 10⁻⁶ Å`). [`thinking_medium` operating points](../../environment/references/s15/summary.csv) · [`n_recycle=0` operating points](../../environment/references/s15/base_schedule_summary.csv) · [`n_recycle=0` coordinate check](../../environment/references/s15/base_schedule_coordinate_check.json)

| Sampler | S15 rate / time (s) | Local `n_recycle=0` clusters / 150 (rate / warm s) | Local `thinking_medium` clusters / 150 (rate / warm s) | `n_recycle=0` clusters/GPU-h incl. AF3 | `thinking_medium` clusters/GPU-h incl. AF3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stochastic TSR | 34.7% / 2.99 | **57 (38.0% / 3.62s)** | 73 (48.7% / 10.06s) | **96.3** | 85.3 |
| Stochastic TSR, prior schedule | 0.0% / 2.97 | **0 (0.0% / 3.63s)** | 0 (0.0% / 9.95s) | 0.0 | 0.0 |
| ODE TSR | 33.3% / 3.04 | **48 (32.0% / 3.64s)** | 70 (46.7% / 10.12s) | 80.9 | 81.5 |
| Full-length SDE | 37.3% / 17.68 | **65 (43.3% / 22.39s)** | 74 (49.3% / 65.27s) | 47.4 | 23.4 |
| Shortened SDE | 0.0% / 2.95 | **0 (0.0% / 3.62s)** | 1 (0.7% / 10.02s) | 0.0 | 1.2 |

Removing the 3× `thinking_medium` recycling overhead accounts for almost the entire 12–14 percentage-point elevation over printed Table S15: Stochastic TSR moves from `48.7%` to `38.0%` (`+3.3 pp` vs `34.7%`), ODE TSR moves from `46.7%` to `32.0%` (`-1.3 pp` vs `33.3%`), Shortened SDE matches `0.0%` (`0/150`), and the 400-to-60-step warm generation time ratio is `22.39 / 3.62 = 6.19×` (vs `5.91×` in the paper). Remaining unknowns relative to the paper's exact numbers are the historical checkpoint commit, random seeds, and H100/H200 batch size used when timing Table S15. [Schedule & multi-seed plot](../../environment/references/s15/s15-schedule-and-repeats.png) · [Original `thinking_medium` plot](../../environment/references/s15/quality-time.png)

## Repeatability and BF16 trajectory sensitivity

1. **Between-campaign `82` vs `73` root cause.** Checking SHA-256 digests confirmed identical weights, model config, and monomer CIFs between the earlier `82/150` run and the `73/150` `thinking_medium` rerun. Slot-by-slot Kabsch alignment shows median native backbone RMSD of `0.052 Å` at `L=100` (with 35/50 slots `< 0.1 Å`, versus `24.79 Å` across distinct seeds), proving identical initial Gaussian noise draws. Under `bf16-mixed` with `torch_deterministic=false`, 1-ULP non-associative kernel reductions accumulate across 180 trunk evaluations (`0.052 Å` median at `L=100`, `0.269 Å` at `L=150`, `0.742 Å` at `L=250`), flipping `2`, `6`, and `10` marginal pass decisions near `1.5 Å`. [Root-cause receipt](../../environment/references/s15/bf16_divergence_root_cause.json) · [Between-campaign check](../../environment/references/s15/default-repeat.json) · [Same-device control](../../environment/references/s15/same-device-repeat.json)
2. **Five-seed paired repeatability.** Across five pre-registered 150-slot seeds (`727000`, `728000`, `729000`, `730000`, `731000`, 1,500 total slots under `thinking_medium`), Stochastic TSR achieved `73, 64, 75, 72, 82` passing clusters (**mean `73.2 / 150 = 48.8% ± 4.3%`**, range `42.7%–54.7%`), while ODE TSR achieved `70, 69, 67, 76, 72` (**mean `70.8 / 150 = 47.2% ± 2.3%`**, range `44.7%–50.7%`). The mean paired advantage of Stochastic TSR over ODE TSR is **`+1.60 pp`** (`+1.4 pp` in Table S15), with single-seed paired differences spanning `-3.33 pp` to `+6.67 pp`. [Multi-seed CSV](../../environment/references/s15/multi_seed_repeats.csv) · [Multi-seed summary JSON](../../environment/references/s15/multi_seed_summary.json)

## Reproduction

```bash
python docs/s15/check_schedules.py
for sampler in stochastic-tsr prior-tsr ode-tsr full-sde short-sde; do
  for length in 100 150 250; do
    # 1. thinking_medium (180 / 1,199 passes)
    python docs/s15/run_cohort.py "$sampler" "$length" generate --assets /workspace/data/assets --root /work/s15 || exit 1
    python docs/s15/run_cohort.py "$sampler" "$length" score --assets /workspace/data/assets --root /work/s15 || exit 1
    # 2. n_recycle=0 base schedule (60 / 400 passes)
    python docs/s15/run_monomer_cohort.py --arm "$sampler" --schedule-mode base_1pass --length "$length" \
      --gen-seed $((727000 + length)) --assets /workspace/data/assets --work "/work/s15_base1pass/$sampler/$length" || exit 1
  done
done
python docs/s15/collect.py /work/s15 /work/s15-report
python docs/s15/plot_results.py /work/s15-report/summary.csv /work/s15-report/quality-time.png
python docs/s15/check_coordinates.py /work/s15 /work/s15-report/coordinate-check.json
python docs/s15/plot_schedule_and_repeats.py --s15-dir environment/references/s15
```

AF3-derived observations are subject to the [AF3 output terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md).
