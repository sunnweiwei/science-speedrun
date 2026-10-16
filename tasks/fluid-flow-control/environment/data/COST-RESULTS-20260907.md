# Final coverage and native-cost decision — 2026-09-07

Frozen resource selection: **hydrogym-public-coverage-v1**, **40 cases (21 two-dimensional, 19 three-dimensional)**; **576 files / 486,282,110,078 bytes (486.282 GB)**.

This is the final decision from this bounded resource/cost audit, not a claim of global optimality across every possible solver configuration. All 62 complete public resource variants have an explicit decision. Unknown cost is not labelled slow.

**Boundary:** source assets and short native reset/control/reset evidence are frozen. Full-episode physics acceptance, the exact evaluator suite, non-MAIA policy adapters and competitive learned-policy scores remain separate unfinished work.

Final packaging, 22:22 UTC: all 576 source files passed complete offline Git/LFS
checksums in the new local release root. Exact selected-file membership and
hard-link identity were checked; both historical release hashes are unchanged.
All 89 task interface/manifest tests passed without skips in an offline R9 CPU
container. No unvalidated `suite.json` was created.

## Included cases — complete short-loop measurements

Times are wall seconds on the stated number of H100s. Samples exclude warm-up. The observed min–max is not a confidence interval. Cold and second-reset costs are measured separately; outer container setup is excluded.

| Case | GB | H100s | Mean s/control | Observed min–max (n) | Cold / reset s | CFD steps/control | Native controls/episode | Estimated episode h |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Cavity_2D_Re7500 | 0.057 | 1 | 0.159 | 0.154–0.162 (5) | 2.50 / 0.20 | 150 | 201 | 0.009 |
| Channel_3D_Retau180 | 0.009 | 1 | 0.104 | 0.104–0.104 (6) | 0.41 / 0.00 | 50 | 100 | 0.003 |
| Cube_3D_Re3700 | 46.191 | 2 | 34.626 | 34.493–34.747 (4) | 733.19 / 54.20 | 1000 | 201 | 1.933 |
| Cylinder_2D_Re1000 | 0.171 | 1 | 0.425 | 0.401–0.446 (5) | 4.20 / 0.55 | 225 | 201 | 0.024 |
| Cylinder_2D_Re200 | 0.069 | 1 | 0.112 | 0.091–0.135 (5) | 3.50 / 0.16 | 87 | 201 | 0.006 |
| Cylinder_2D_Re3900 | 0.191 | 1 | 0.687 | 0.636–0.708 (5) | 4.44 / 0.85 | 500 | 201 | 0.038 |
| Cylinder_3D_Re200 | 23.164 | 1 | 42.610 | 41.826–43.676 (4) | 653.76 / 61.60 | 575 | 201 | 2.379 |
| Cylinder_3D_Re3900 | 41.225 | 2 | 88.976 | 86.773–91.498 (4) | 761.36 / 112.64 | 1080 | 201 | 4.968 |
| Kolmogorov_2D | 0.000 | 1 | 3.455 | 3.415–3.489 (6) | 4.66 / 3.90 | 10000 | 1000 | 0.960 |
| NACA0012Gust_2D_Re1000_AOA20 | 0.051 | 1 | 0.092 | 0.050–0.122 (5) | 2.57 / 0.11 | 60 | 201 | 0.005 |
| NACA0012Gust_2D_Re1000_AOA40 | 0.051 | 1 | 0.100 | 0.072–0.124 (5) | 2.49 / 0.11 | 75 | 201 | 0.006 |
| NACA0012Gust_2D_Re100_AOA20 | 0.051 | 1 | 0.092 | 0.052–0.121 (5) | 2.62 / 0.10 | 75 | 201 | 0.005 |
| NACA0012Gust_2D_Re100_AOA40 | 0.051 | 1 | 0.112 | 0.088–0.133 (5) | 2.61 / 0.13 | 75 | 201 | 0.006 |
| NACA0012Gust_3D_Re10000_AOA12 | 46.558 | 2 | 38.954 | 38.844–39.145 (4) | 1986.54 / 67.12 | 150 | 201 | 2.175 |
| NACA0012Gust_3D_Re1000_AOA20 | 36.490 | 1 | 18.173 | 17.620–18.727 (2) | 790.90 / 49.67 | 160 | 201 | 1.015 |
| NACA0012Gust_3D_Re1000_AOA40 | 34.949 | 1 | 20.055 | 19.749–20.451 (4) | 779.56 / 51.59 | 175 | 201 | 1.120 |
| NACA0012Gust_3D_Re100_AOA20 | 23.608 | 1 | 5.560 | 5.338–5.858 (5) | 539.21 / 25.44 | 75 | 201 | 0.310 |
| NACA0012Gust_3D_Re100_AOA40 | 23.494 | 1 | 6.255 | 5.955–6.488 (5) | 531.36 / 26.55 | 90 | 201 | 0.349 |
| NACA0012Gust_3D_Re50000_AOA12 | 46.367 | 2 | 35.869 | 35.729–36.006 (4) | 1981.41 / 63.89 | 140 | 201 | 2.003 |
| NACA0012_2D_Re1000_AOA20 | 0.051 | 1 | 0.218 | 0.195–0.250 (5) | 2.70 / 0.25 | 170 | 201 | 0.012 |
| NACA0012_2D_Re1000_AOA40 | 0.051 | 1 | 0.350 | 0.330–0.393 (5) | 2.59 / 0.38 | 400 | 201 | 0.020 |
| NACA0012_2D_Re100_AOA20 | 0.051 | 1 | 0.442 | 0.404–0.487 (5) | 2.55 / 0.51 | 375 | 201 | 0.025 |
| NACA0012_2D_Re100_AOA40 | 0.051 | 1 | 0.449 | 0.431–0.473 (5) | 2.51 / 0.48 | 375 | 201 | 0.025 |
| NACA0012_3D_Re1000_AOA20 | 36.490 | 1 | 41.762 | 39.795–43.294 (4) | 826.86 / 76.16 | 375 | 201 | 2.332 |
| NACA0012_3D_Re100_AOA20 | 23.608 | 1 | 34.293 | 33.927–34.771 (4) | 530.08 / 56.91 | 500 | 201 | 1.915 |
| NACA0012_3D_Re100_AOA40 | 23.494 | 1 | 33.189 | 32.611–33.622 (4) | 538.81 / 53.89 | 500 | 201 | 1.853 |
| Nozzle2D_coarse | 0.008 | 1 | 0.904 | 0.900–0.907 (5) | 26.04 / 0.58 | 100 | 667 | 0.167 |
| Nozzle2D_fine | 0.025 | 1 | 2.918 | 2.878–3.042 (11) | 17.52 / 0.57 | 200 | 667 | 0.541 |
| Pinball_2D_Re100 | 0.076 | 1 | 0.228 | 0.210–0.244 (5) | 3.40 / 0.30 | 200 | 201 | 0.013 |
| Pinball_2D_Re150 | 0.047 | 1 | 0.208 | 0.193–0.241 (5) | 3.57 / 0.26 | 150 | 201 | 0.012 |
| Pinball_2D_Re30 | 0.076 | 1 | 0.394 | 0.368–0.416 (5) | 3.13 / 0.43 | 270 | 201 | 0.022 |
| Pinball_2D_Re75 | 0.076 | 1 | 0.272 | 0.253–0.295 (5) | 3.35 / 0.34 | 200 | 201 | 0.015 |
| Pinball_3D_Re100 | 8.342 | 1 | 10.126 | 9.856–10.273 (5) | 243.63 / 17.85 | 195 | 201 | 0.565 |
| Pinball_3D_Re150 | 8.342 | 1 | 10.642 | 10.478–10.930 (5) | 250.06 / 18.12 | 175 | 201 | 0.594 |
| Pinball_3D_Re30 | 8.342 | 1 | 15.947 | 15.536–16.245 (5) | 247.85 / 21.20 | 270 | 201 | 0.890 |
| Pinball_3D_Re75 | 8.342 | 1 | 12.157 | 11.358–12.546 (5) | 241.38 / 19.40 | 210 | 201 | 0.679 |
| Sphere_3D_Re300 | 19.965 | 1 | 28.752 | 28.612–28.919 (5) | 131.57 / 44.50 | 825 | 201 | 1.605 |
| SquareCylinder_2D_Re1000 | 0.087 | 1 | 0.382 | 0.357–0.430 (5) | 3.04 / 0.40 | 250 | 201 | 0.021 |
| SquareCylinder_2D_Re200 | 0.070 | 1 | 0.096 | 0.062–0.124 (5) | 3.43 / 0.13 | 110 | 201 | 0.005 |
| SquareCylinder_3D_Re200 | 25.941 | 1 | 65.216 | 64.685–66.128 (4) | 769.75 / 84.67 | 750 | 201 | 3.641 |

**Episode column is a short-rate extrapolation, NOT an executed full episode.** It excludes initialization, reset, policy inference, learner work, communication contention and possible action-dependent changes. Summing one such episode per selected case gives about **43.34 GPU-hours of integration** before those costs; dividing by eight is only an ideal packing lower bound, not a measured full-suite wall-time guarantee.

Native horizon details were checked against the pinned code: MAIA uses `iter > 200`, hence 201 actions (a reviewed TorchRL time-limit protocol may instead use 200); channel time advances by 50 CFD steps/action and terminates at 5000, hence 100 actions, not 5000; Kolmogorov uses 1000 actions; 2D nozzles terminate at physical time 0.01 with 667 actions at their native fixed dt. These are cost profiles, not an implicit final scoring-horizon choice.

## Every excluded complete resource

| Case | GB | Reason | Evidence and interpretation |
| --- | ---: | --- | --- |
| Cavity_2D_Re7500_fine_FD | 2.461 | evaluation_protocol_unbound | Four-CPU one-CFD-substep probes ran, but the accepted full control/episode protocol remains unbound. Not measured slow and not excluded because large; matching MAIA task families are retained. |
| Cavity_2D_Re7500_medium_FD | 0.724 | evaluation_protocol_unbound | Four-CPU one-CFD-substep probes ran, but the accepted full control/episode protocol remains unbound. Not measured slow and not excluded because large; matching MAIA task families are retained. |
| Cube_3D_Re300 | 14.090 | source_budget_tradeoff | Passed: 43.44 s/control on one H100 (4 warm samples). Including it with the other selected cases would require 500.372 GB, above the hard cap. Prefer the validated Re3700 cube while preserving the paired wing regimes. This is a budget/value choice, not a runtime failure or a claim that Re300 is too slow. |
| Cylinder_2D_Re100_fine_FD | 1.174 | evaluation_protocol_unbound | Four-CPU one-CFD-substep probes ran, but the accepted full control/episode protocol remains unbound. Not measured slow and not excluded because large; matching MAIA task families are retained. |
| Cylinder_2D_Re100_medium_FD | 0.125 | evaluation_protocol_unbound | Four-CPU one-CFD-substep probes ran, but the accepted full control/episode protocol remains unbound. Not measured slow and not excluded because large; matching MAIA task families are retained. |
| Cylinder_2D_Re200_fine_FD | 1.174 | evaluation_protocol_unbound | Four-CPU one-CFD-substep probes ran, but the accepted full control/episode protocol remains unbound. Not measured slow and not excluded because large; matching MAIA task families are retained. |
| Cylinder_2D_Re200_medium_FD | 0.119 | evaluation_protocol_unbound | Four-CPU one-CFD-substep probes ran, but the accepted full control/episode protocol remains unbound. Not measured slow and not excluded because large; matching MAIA task families are retained. |
| Cylinder_3D_Re1000 | 41.225 | compatible_cost_not_established | Native single-GPU initialization aborted before valid controls. No unexplained MPI/device abort is labelled OOM or slow. This bounded audit prioritizes successfully validated two-GPU high-Re gust wings/cylinder/cube and retains lower-Re family counterparts; this exact variant needs a further compatible capacity test. |
| DRA2303Jet_3D_Re400000_Ma02 | 93.473 | native_solver_configuration_unresolved | The structured finite-volume source configuration does not match the exercised LB constructor (missing reductionFactor and other required bindings). No valid control timing; not a measured slow solver. Requires a separately validated native launch route. |
| NACA0012_3D_Re10000_AOA12 | 46.558 | compatible_cost_not_established | Native single-GPU initialization aborted before valid controls. No unexplained MPI/device abort is labelled OOM or slow. This bounded audit prioritizes successfully validated two-GPU high-Re gust wings/cylinder/cube and retains lower-Re family counterparts; this exact variant needs a further compatible capacity test. |
| NACA0012_3D_Re1000_AOA40 | 34.949 | measured_high_integration_cost | One warm sample is 147.03 s/control on one H100 (1300 CFD substeps/action), about 8.2 h for 201 controls before overhead. Rough screening estimate, not a sustained rate. Retain the corresponding gust case and the cheaper no-gust AOA20 case. |
| NACA0012_3D_Re200000_AOA0 | 174.526 | local_capacity_and_runtime_unverified | 174.53 GB NEK wing has a public 4096-MPI-rank configuration versus this host's 208 CPUs. A compatible local decomposition/runtime was not validated. Neither launched blindly nor reported as a measured slow case. |
| NACA0012_3D_Re50000_AOA12 | 46.367 | compatible_cost_not_established | Native single-GPU initialization aborted before valid controls. No unexplained MPI/device abort is labelled OOM or slow. This bounded audit prioritizes successfully validated two-GPU high-Re gust wings/cylinder/cube and retains lower-Re family counterparts; this exact variant needs a further compatible capacity test. |
| Nozzle3D_coarse | 2.880 | measured_high_episode_cost | Correctly restarted two-H100 probe completed with field response and second reset: warm controls 234.48 and 234.21 s. Native CFL-adaptive 0.01 s horizon extrapolates to about 16 wall-hours / 32 GPU-hours. Explicit organizer eight-actuator diagnostic, not paper reproduction. Keep both 2D nozzles; source size alone was not the issue. |
| Nozzle3D_fine | 14.400 | distributed_restart_binding_unverified | Complete distributed resources but restart/decomposition and actuator binding remain unvalidated. Its speed was not established; the validated coarse 3D case already has very high episode cost. No claim that this unrun variant was measured slow. |
| Pinball_2D_Re100_fine_FD | 0.348 | evaluation_protocol_unbound | Four-CPU one-CFD-substep probes ran, but the accepted full control/episode protocol remains unbound. Not measured slow and not excluded because large; matching MAIA task families are retained. |
| Pinball_2D_Re150_fine_FD | 0.348 | evaluation_protocol_unbound | Four-CPU one-CFD-substep probes ran, but the accepted full control/episode protocol remains unbound. Not measured slow and not excluded because large; matching MAIA task families are retained. |
| Pinball_2D_Re30_fine_FD | 0.348 | evaluation_protocol_unbound | Four-CPU one-CFD-substep probes ran, but the accepted full control/episode protocol remains unbound. Not measured slow and not excluded because large; matching MAIA task families are retained. |
| Sphere_3D_Re3700 | 34.443 | measured_high_integration_cost | One warm sample is 171.26 s/control on one H100 (2400 CFD substeps/action), about 9.6 h for 201 controls; the bounded probe did not finish cleanup. Not a precise sustained rate. Retain Sphere Re300 plus higher-Re cube/cylinder/wing cases. |
| SquareCylinder_3D_Re1000 | 48.646 | compatible_cost_not_established | Native single-GPU initialization aborted before valid controls. No unexplained MPI/device abort is labelled OOM or slow. This bounded audit prioritizes successfully validated two-GPU high-Re gust wings/cylinder/cube and retains lower-Re family counterparts; this exact variant needs a further compatible capacity test. |
| SquareCylinder_3D_Re3900 | 48.646 | compatible_cost_not_established | Native single-GPU initialization aborted before valid controls. No unexplained MPI/device abort is labelled OOM or slow. This bounded audit prioritizes successfully validated two-GPU high-Re gust wings/cylinder/cube and retains lower-Re family counterparts; this exact variant needs a further compatible capacity test. |
| TCFmini_3D_Re180 | 0.020 | solver_restart_mismatch | Installed binary has 128 elements versus YAML 256, and restart dt=0.002 versus YAML 0.01. Fast CPU smoke timing does not certify the intended configured task. |

Genuinely missing/marker-only/mismatched public resources are listed separately in [release.json](release.json); they are not among the 62 complete variants above.

## Corrections and provenance

- Two-GPU MAIA needs one solver rank per distinct CUDA device. Earlier allocated-two-GPU attempts actually shared one device; they are explicitly invalidated, retained for audit, and excluded from valid timing evidence. Qualifying reruns have pre-exec CUDA UUID receipts and process-to-GPU checks. The reusable launcher now handles explicit rank binding.
- Fine 2D and coarse 3D nozzle source JSONs omit the restart-enable flag. The runtime-only binding enables their supplied restart without editing source bytes or interpolating grids. Fine 2D then shows nonzero flow, evolving fields and signed thrust response. Coarse 3D also integrates, but its two warm samples are about 234 seconds/control on two H100s. Its 0.01-second CFL-adaptive horizon extrapolates to approximately 16 wall-hours / 32 GPU-hours. The diagnostic 8-actuator count is an explicit organizer choice, not a recovered paper setting.
- Kolmogorov uses the pinned official `minimize_tke` example with alpha=1, Re200, 64² and float64; not the asset-YAML Re40 or paper Re250 profile. Channel uses pinned class defaults (72³, 24 actions, dt=0.0002), not its differing asset YAML.
- MAIA diagnostic sensors are not certified paper/evaluation sensors. Grids, original restart files, dt and native actuation intervals were not reduced to improve timing.

Manifest SHA256: `365e9531671ae4eb24491e880f7e6c1d9768013b5da2bb389df9342525ca1896`.

Pinned resource revision: `cf5ef6748443ae72e9647f398487e98d421a7243`. Pinned code revision: `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`.

Machine-readable [selection](selection-coverage-v1.json) and [evidence](cost-evidence-coverage-v1.json) include effective profiles, raw samples, source sizes, immutable image IDs, binding receipts and raw-record hashes. Raw logs are preserved on the development host under `/mnt/workspace/hydrogym/reproduction/coverage-expansion-20260907` and the earlier `case-cost-audit-20260907` directory.

The 500 GB cap is the conservative sum of all named selected source files; no compression or duplicate-file discount. Training and evaluation share one copy. Images, trajectories, checkpoints, scratch and old archives are outside that source cap. Historical releases and unrelated training processes were preserved. No new agent session, model training, external upload or push is part of this audit.
