# Observation-grid numerical control

On APE2D, apply the released Exponax equation to the last public 256 × 256 frame, using the published interval, substeps and physical coefficients. The reference data uses 2048 × 2048 generation followed by 8 × 8 averaging. This control treats the observed values as its grid state. It does not use a learned model, fitted parameters, training data or hidden initial conditions. [Original APE2D generator](https://github.com/tum-pbs/pde-transformer/blob/850e09d352487922e4a853ee90a2624717769cc7/pdetransformer/data/simulations_apebench/simulation_setups_2d_low_res.py)

On PDEArena smoke, restore the 128 × 128 scalar and staggered velocities from the last public frame. PhiFlow performs four steps of 1.5 simulator time units per saved frame. On the two The Well datasets the comparison repeats the last observed frame. The resulting four-group score describes this explicit numerical/persistence policy; group scores are available separately. [Original PDEArena simulator](https://github.com/pdearena/pdearena/blob/78a8b03d50115d8bb24ce9f04efa5b920fcd4369/pdedatagen/navier_stokes.py)

## Reproduce

Stage the 1,085 canonical test cases with the task case builder and four context frames. Set `PUBLIC`, `TRUTH`, and `OUTPUT` to the staged public input, hidden truth, and prediction directories. Use a fresh `OUTPUT` for a new suite. Each invocation below reads only public inputs. Set `APE_PY` to Python 3.10 with Exponax 0.1.0, JAX/JAXlib 0.4.35, Equinox 0.11.10 and h5py. Set `PHI_PY` to Python 3.10 with PhiFlow 2.1.0, NumPy 1.23.5, SciPy 1.10.1 and h5py 3.10.0.

```bash
"$APE_PY" coarse_solver.py --group ape2d --input "$PUBLIC" --output "$OUTPUT"
"$PHI_PY" coarse_solver.py --group pdearena --input "$PUBLIC" --output "$OUTPUT"
"$PHI_PY" coarse_solver.py --group well --input "$PUBLIC" --output "$OUTPUT"
"$PHI_PY" ../../../environment/evaluation/score.py \
  --input "$PUBLIC" --truth "$TRUTH" --output "$OUTPUT" --report "$OUTPUT/report.json"
```

Each group writes a filtered `index.<group>.json` and failure list. The task scorer uses the original full index, including any failed cases, and first averages datasets within each group. Checkpoint and frontier-model comparisons use the same frozen published targets. For this pilot, refined PDEArena targets, training-shift estimates and rollout costs are reported separately.
