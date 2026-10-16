# Active completion audit — 2026-09-06

The user requires paper-performance reproduction, constructed baseline resources,
and runnable baselines in the task's starter code. Do not mark this complete on
the strength of one environment, a solver smoke test, a fixed-control diagnostic,
an exported checkpoint, a passing interface test, or a partial data inventory.
No author contact and no AutoResearch agent launch are authorized in this phase.

## Required evidence still outstanding

1. Reproduce the paper's demonstrated baseline performance, starting with but not
   ending at 2D Pinball. Compare PPO/DDPG/TD3 learning curves and frozen policies
   across three independently trained seeds. Continue the 3D Pinball, cavity,
   cylinder, gust-airfoil and differentiable PPO/GPPO demonstrations, with explicit
   public-source reconstruction choices where exact settings are absent.
2. Establish paper-to-resource/solver/protocol coverage for the intended full
   HydroGym task, not a one-case replacement. Downloaded directories and YAML
   counts are not counts of validated benchmark cases. The paper tables contain
   scenarios with no matching published resource directory; do not silently omit
   those or label coverage complete.
3. Baselines must be executable from the task development image's starter folder,
   with their source, configurations, independent inference/evaluation and real
   measured result metadata. Reference specialists are not a valid shared-policy
   submission. The eventual shared-policy starter and external evaluator need
   native CFD testing, not just CPU synthetic tests.
4. Construct the actual local data release, resource checksums, baseline resources
   and reproducible materialization interface. Keep full-suite certification draft
   until the corresponding physical/protocol/reference checks really succeed.

## Evidence this goal turn

- First 500-episode batch does NOT match paper performance. Completed seeds are
  in `hg-public-pinball2d-{ppo,ddpg,td3}-s{42,43,44}-20260905` on the data disk.
  All nine training-to-independent-final/best workflows have completed successfully.
  Their 207 archived files, including weights, raw traces and frozen source, are in
  `data/baseline-results/public-pinball-20260905`; this is explicitly an archive of
  non-matching candidate results, not accepted paper baseline anchors.
- PPO's three final-seed average full-window drag reduction was only 10.14%; TD3
  50.12%. All three final DDPG seeds averaged 62.61%. TD3 seed 44's selected best
  attained 85.82% full-window / 91.19% last-half reduction but RMS lift remained
  0.7764 and raw return -198.49. One favorable seed/window is not reproduction.
- Six new documented seed-42 training recipes are running, 100,000 controls each:
  `hg-pinball2d-{ppo-scaled,ppo-narrow,ppo-author,td3-slow,td3-scaled,ddpg-slow}-s42-20260906`.
  Their native physics/reward is unchanged; constant training reward scaling,
  actor/critic LR and PPO exploration/author-example choices are recorded in JSON.
  All recipes passed actual optimizer-update and checkpoint-reload CPU tests.
- `hg-pinball3d-zero-long-20260906` measures a continuous 1,000-action uncontrolled
  trajectory to investigate the 3D mean-drag gap. Its diagnostic horizon is
  explicitly recorded, not presented as the paper's 200-step evaluation.
- Full pinned public-resource download and Git/LFS hash verification completed
  successfully at approximately 09:21 UTC. `materialized.json` lists all 89
  verified directories. The completed script/log are under
  `/mnt/workspace/hydrogym/reproduction/materialize-full-20260906.*`.
  Catalog inventory: 89 directories, 66 YAML configs, 1,254,095,281,221 bytes.
  Resource completion does not certify scientific case coverage: missing paper
  cases and native solver/reward differences still require reconstruction.
- `export_policy.py` packages trained reference actors as self-contained TorchScript
  resident policies. TD3 seed 44's best was exported to
  `data/baseline-candidates/Pinball_2D_Re100/td3-s44-public-20260905` (development,
  NOT paper-matched). It agrees with the original actor on 64 test observations
  at batch size one (max absolute error 5.96e-8), and served in an offline container
  with no training directory, author source or original checkpoint mounted.
  Full 200-action native CFD replay via production JSON-lines transport completed:
  return -198.4936591981 versus original -198.4936679575 (difference 8.76e-6).
  Relocated package SHA-256 is
  `dd1369c54b9dcc33d012db78826decc2517723ca4e2cda9957494eddf08839f5`.
- Development image's second build succeeded at 09:22 UTC, image
  `ae9dcd0a1cac5a2b3431e5b45f34410ed4ba94bf311335865fd73e57c95063b0`.
  It installs pinned TorchRL/author references and copies manual baselines into
  `/workspace/baseline/starter/paper_baselines`. Further edits add explicit native settings
  and cylinder reconstruction; build/validate those before declaring starter parity.
- All 22 CPU task tests passed in the declared ML runtime, including standalone
  relocated TorchScript inference and shared starter train/resume/publication.
  Three additional native-setting tests pass on host; full ML rerun is pending.
- `Cylinder_2D_Re3900` is now accepted by the manual launcher, using paper 600
  substeps/200 controls, native zero-net-flux jets and author sensor defaults.
  This extension has not yet passed a native CFD run. See CYLINDER_RECONSTRUCTION.md.
- Cavity source audit uncovered an additional reward mismatch: native current
  implementation sums absolute deviations; Supplementary Eq. 36 uses squared,
  target-standardized deviations. Do not claim paper parity for the former.
- Latest validated development image is
  `70c8b32e29a016255b23b90602f0f5215061e14f04e1e302e52b6336b093660b`.
  All 25 task CPU tests and all nine learner/recipe update+reload checks passed.
  Its in-image 1,200-step PPO training plus independent final/best replay passed.
  Cylinder zero-control 200-action check also passed: CD=1.62350477,
  RMS CL=1.34621355, raw return=-566.91621902. This is not learned performance.
  The two-case shared-policy train/resume/relocated-CFD-replay check completed
  successfully in `hg-shared-native-check-20260906`: one 800-step shared model,
  two 200-action native evaluations after relocation, unchanged package hash
  `1ae0d56e29d0d5d789eb6a80ae67d2f02812988fad744708210e7fc4be1f3fe2`.
- JAX native CFD TKE gradient passed an actual central-difference comparison on
  CPU (50.8 seconds). `jax_public.py` now launches the public PPO factory with
  a documented related-author mixing reward (TKE - 75*L1 control), Re250 and paper
  table's 64x64 grid. The first 400-transition pilot became nonfinite in its second
  rollout and was stopped, with failure evidence retained. Instrumented 80-step
  FP32 and FP64 diagnostics both reproduced nonfinite second-rollout values.
  Source tests established no-op native dealiasing and a missing control-curl
  factor relative to the older author solver. Independent opt-in `dealias` and
  `dealias-curl` diagnostics are running, alongside a full-action-trace original
  run. See JAX_RECONSTRUCTION.md; no stable learning/GPPO claim has been made.
- Public base image lacks pyarrow, required by native cavity baseline-state
  reading. A separate image `ai4sci-hydrogym-task:cavity-20260906` successfully
  built with pyarrow 19.0.1, ID
  `c1a6eb274f6edce2b425dca0f980c0c057e661ceb078baf3abb0c4683e213a1f`.
  The first cavity measurement exposed a missing native YAML `maia.U_inf`; that
  failed MPMD job was stopped, logs/source preserved. The v2 diagnostic derives
  lattice U_inf from native Ma, records that reconstruction, and is collecting
  1,000 uninterrupted zero-control samples on GPU 1. This is NOT 1,000 cycles.
  Output: `hg-cavity2d-uncontrolled-probe-v2-20260906`; raw samples remain finite.
  New JAX/cavity scripts run from explicit host mounts; this image's build-context
  snapshot predates those additions. Rebuild/test final starter before release.
- `summarize_public.py` now refuses incompatible zero-control protocols and
  groups only distinct training seeds with identical full learner configurations
  and physical settings. Different recipes must never be pooled as seed repeats.
  The regression test passes. Six live recipe trials are approximately 15k–19k
  controls at 10:06 UTC; best TD3-scaled at 12k had 85.22% full-window reduction,
  but degraded at 16k and still had high lift. No convergence claim is justified.

Source worktree is dirty/untracked from previous task construction. Preserve it.
No commits, GitHub pushes, HF uploads, author messages or unrelated container
changes were made. Runtime artifacts stay on `/mnt/workspace/hydrogym`.

## Latest correction audit — approximately 10:25 UTC

- Four organizer JAX diagnostics omitted `PYTHONPATH` and imported the image's
  older editable CFD copy. They are excluded; see
  `jax-source-exclusions-20260906.json` in the reproduction root. Original pinned
  gradient and FP32/FP64 failure checks are unaffected, as are MAIA experiments.
  A wrong-source negative test passes, and new `jax_source.py` now checks actual
  imported file hashes. Three `hg-jax-pinned-{trace,dealias,curl}-f32-20260906`
  80-transition diagnostics are active on CPU 160–207 with verified source hashes.
- `hg-jax-pinned-trajectory-gradient-20260906` passed a two-control, corrected
  Re250/64x64 physical-gradient/finite-difference check: analytical directional
  derivative 12.54131, numerical 12.41588 with epsilon 1e-4 (~1% difference),
  total 104.5 seconds. This is not GPPO training or long-horizon certification.
- Cavity Re4140's 1,000-sample pilot passed: dominant period 6.7568 controls,
  74 peaks in its last 500 samples; not 1,000 cycles. A derived Re4200 binding
  preserves all public assets and records the two Reynolds-field changes.
  `hg-cavity2d-re4200-calibration-20260906` is running 16,000 controls on GPU 1.
  The solver log confirms Re4200; its source file matches the pinned MAIA hash.
  Need inspect >=1,000-cycle coverage/mean drift, create a relaxed Re4200 restart
  and a tested Eq.36 reward adapter before attempting paper-like cavity training.
- A temporary read-only Python source-inspection process inside the cavity
  container hung during singleton MPI initialization. Only that diagnostic and
  its own two MPI helper processes were terminated; the actual paired CFD worker
  continues normally. Use file hashing, not importing MAIA from an extra singleton
  process in a running MPI container, for subsequent version checks.

## Active work — approximately 10:44 UTC

- Pinned JAX diagnostics now provide a causal result: original CFD fails on
  the second recorded action; the exact same actions remain finite with either
  dealiased mode. Both corrected PPO 80-step checks completed with real parameter
  updates. Independent 10-step zero control returned +29.730408; the early
  deterministic PPO checkpoint returned only +1.628174. Do not conflate these
  deterministic replays with the paper's stochastic training-return curve.
- `hg-jax-ppo-paper-budget-v2-20260906` runs corrected public PPO for 5,000
  controls / 500 episodes on CPU 192–207. First 40 controls/80 optimizer updates
  completed and atomically published `params-0000040.msgpack`, with SHA
  `134c0b6a5d411c85c18f8ac13a87eb560f24a21399b95d49448e3de3cab5bc7f`.
  It has ordered publication callbacks and source guards. A previous startup was
  stopped before publishing to enforce ordered callbacks; use v2, not that run.
- Fixed physical Pinball controls [0,-0.375,0.375] achieved raw return -29.77658,
  full-window CD=0.0952513, last-half CD=-0.0136796 and RMS CL=0.0611587, much
  closer to the paper plot than the 0.25 fixed counter-rotation. This is a manual
  physics diagnostic, NOT RL reproduction. It motivates author-example max=0.5.
- Stopped only `ppo-scaled` and `ppo-author` max=0.25 ablations; all partial
  artifacts remain intact, not pooled with completed runs. Other four recipes
  continue on GPUs 2/4/5/6. See `control-limit-reallocation-20260906.json`.
- Three max=0.5 short checks are running: PPO-narrow GPU 0, DDPG-slow GPU 3,
  TD3-scaled GPU 7 (temporarily shares with 3D diagnostic). Queue
  `queue-a050-full-20260906.sh`, tmux `hg-a050-full-20260906`, starts full-budget
  PPO on 0 after its check passes, and TD3 on 3 after that GPU's DDPG check ends
  and TD3's own check passes. Full runs use the successful pilot's frozen source.
- Cavity Re4200 16,000-control calibration still running on GPU 1. Its initial
  public Re4140 restart is explicitly documented. Need inspect cycle coverage and
  stationarity, then generate a relaxed Re4200 restart for later learning resets.
- Current task development image is the earlier c1a6... build with pyarrow but
  without the latest host-mounted JAX/cavity source additions. Final starter
  image rebuild/testing remains required; do not claim all new baselines are
  already included in that image.

## Active work — approximately 11:08 UTC

- All three max=.5/1200-control Pinball checks finished and fresh final/best
  replays passed. Full PPO-narrow/.5 is active on GPU0 (~4400 controls) and
  TD3-scaled/.5 on GPU3 (~3000), using their pilot's frozen source. Early scores
  are still poor, not matched performance. Three .25 trials remain on GPUs2/4/5;
  TD3-scaled's 28k replay is -213.18, best -207.676, still not converged.
- Stopped DDPG-slow/.25 on GPU6 after its 32k replay remained worse than zero,
  preserving all ~34k partial artifacts. `cavity-author-reallocation-20260906.json`
  records the replacement; it is not a completed seed and must not be pooled.
- New cavity source audit recovered the old-to-new length scale (20:1), making
  the public 34-probe example and its 68-entry normalization/target vectors
  usable after explicit coordinate conversion. See CAVITY_RECONSTRUCTION.md.
  SHA-verified `cavity_author.py` reads exact public vectors. Do not silently
  treat them as the paper's final figure metadata.
- `hg-cavity2d-author34-zero-check-20260906` is active on GPU7 (shares 3Dzero),
  1000 samples/author Eq.36 and private restart writer. At ~600 controls, rewards
  remain around -100; no paper match. `hg-cavity2d-author34-calibration-20260906`
  starts 16000 controls on GPU6 with a final restart writer. The earlier 25-probe
  16000 calibration remains on GPU1 (~8700); their statistics are not compatible.
- Restart writer check exited0, actual Re4200/Ma.1/step605002 confirmed using
  netCDF4. Input writer overwrites restart_.Netcdf, so private copies are required.
  Original resource SHA remains intact. Manifest `cavity-restart-605002-20260906.json`.
- New cavity_reward/calibrate/restart/train modules; four formula/binding tests
  and three real learner-update/reload synthetic cases pass. Public learner now
  supports configured horizons and checks preprocessing/target/restart parity.
  Existing running Pinball learners use frozen older source, unaffected.
  Native `hg-cavity2d-eq36-restart-check-20260906` performs 2x100 zero replays on
  GPU7 before launching cavity learning. No full cavity PPO/DDPG/TD3 run yet.
- JAX PPO500episodes is finite at 240/5000 global controls, 480 optimizer updates,
  atomic immutable checkpoints. Last4 stochastic returns -490,-805,-392,-608.
  It is still a corrected, reconstructed setup, not a paper performance match.
- Current Docker task image is still c1a6...; new source is host-mounted and
  frozen per run. Final task image build/test and full benchmark coverage remain.

## Active work — approximately 11:49 UTC

- Image r2 (`547dc8a...`) and r3 (`f70fa782...`) are built. r3 passed 25 task
  tests, five cavity formula/calibration tests, three cavity learner checks,
  nine recipe checks and actual pinned-source/normalization imports. They
  predate the latest JAX CUDA-loader/callback fixes and NACA diagnostic; final
  image packaging/tests remain. Do not assume latest host scripts are in r3.
- Corrected GPU JAX PPO80 passed actual updates and independent replay (return
  13.623970, below zero29.730408). Full GPU5000 started11:35 using frozen source.
  `JAX_PLATFORMS=cuda,cpu` and `jax_runtime.sh` are required: MAIA's NVHPC
  library environment mixes incompatible CUDA libraries; CUDA-only backend
  disables publication callbacks. CPU5000 remains running as a same-seed
  backend ablation. GPPO remains unimplemented, no paper performance match.
- Cavity25-probe16000 completed, last8000 cover~1203 periods/1202 peaks,
  max blockwise mean drift .002234 standard deviations. Frozen calibration:
  `cavity-calibration-reconstructed25-20260906.json` in reproduction root.
  This is NOT author34-compatible or paper-certified; no relaxed restart was
  saved by that older run. Author34 calibration remains active on GPU6.
- Author34 zero1000 and checked-restart 2x100 replay both exited0. Repeated
  returns at the checked605002 restart are exactly -12585.708878.
  Persistent tmux `hg-cavity-and-authornorm-20260906` created a manifest from
  the author34zero1000 actual final restart, then launched in-image r2 cavity
  PPO400 on GPU1 with final/best fresh-process evaluation. Its initial random
  policy return is -18184.8775. No full-budget cavity baseline has run yet.
  The queue next runs Pinball author-cached-normalization PPO1200 and, only if
  it passes, full100000 on GPU1. Public34 and reconstructed25 stats must never
  be interchanged. Do not duplicate the queue's jobs or manifest files.
- Pinball public author normalization is now SHA-guarded, including original
  y-major 7x7 probe order and98-entry mean/std. Earlier manual runs used U_inf
  scaling and a different probe order; they remain documented separate trials.
- NACA3Dgust native.py smoke/zero support and a reconstruction audit added,
  but no actual NACA solver launch yet. Its73M-cell/34GB restart and two-solver
  configuration require a real memory/launch check before any learned baseline.
- All nine original full Pinball seeds remain completed-but-not-paper-matched;
  five newer full .25/.5 experiments continue. Full suite/backend coverage is
  still incomplete. No goal completion, author messages, commits or pushes.

## Active work — approximately 12:22 UTC (supersedes queue/image status above)

- R4 image built: `ai4sci-hydrogym-task:baseline-r4-20260906`, SHA
  `aa1ab1da95c5cf5d463ad095b696cb8f8c74ca0ba92c93b4c7b404cbdb364439`.
  `hg-r4-image-tests-v2-20260906` passed25 task tests with only tests mounted;
  implementation symlinks point to image-packaged code. Also five cavity
  formula/calibration tests, three real synthetic cavity learner checks,
  nine Pinball recipe checks and actual source imports passed, no skips.
  `hg-r4-image-jax-qr-20260906` independently passed GPU QR/clean-loader checks.
  The earlier `hg-r4-task-unit-20260906` discovered zero tests and is NOT evidence.
  R4 contains MPI error handling, cavity .02/.075 choices, JAX bootstrap and
  full-policy derivative diagnostic. It predates the latest JAX optimizer-side
  reward-scale option; that option is currently host/frozen-source only.
- Cavity .075 PPO became nonfinite on first training episode/control22 (all68
  raw observations). The first diagnostic hung in finally-close; py-spy confirmed
  MPI timestep-send deadlock. Error-logging rerun reproduced the nonfinite state
  and exited promptly. `cavity_train.py` and future `native.py` jobs now abort
  paired MPI ranks on errors before normal close, preserving tracebacks.
  Original .075 failures remain preserved and excluded.
- Current-resource bound .02 PPO400 passed actual updates, final/best fresh
  replays, TorchScript export and external resident-JSONL/CFD replay. Final
  return -12771.518659524363 matched EXACTLY across native and exported replay;
  best was initial-8542.111678. Not a learned performance success. See
  `hg-cavity2d-ppo-nativebound-check-20260906`, exported package
  `hg-cavity2d-ppo-nativebound-package-20260906`, and external replay
  `hg-cavity2d-export-replay-check-20260906` in reproduction root.
- Pinball author-normalization PPO1200 passed final/best replay, final-818.3754,
  best-750.8380. It was temporarily paused during cavity checks to avoid very
  slow GPU-context sharing, then unpaused; it is finished now. Full run
  `hg-pinball2d-ppo-authornorm-200k-s42-20260906` started12:10:06 onGPU1.
  Budget200000 follows executable author collector YAML, despite its unused
  num_episodes500 field. This is1000 actual200-action episodes/500 batches,
  not silently relabelled as the paper's500-episode axis. Older100k runs unchanged.
- Stopped inferior U_inf PPO/.25 onGPU2 at~40k for NACA. Also stopped U_inf
  PPO/.5 onGPU0 at~10600 and TD3-slow/.25 onGPU4 at~43200. All partial outputs
  preserved, not completed seeds. Named reallocation JSON audits in run root.
  GPU0 now only JAX; simultaneous MAIA/JAX made one200-control Pinball episode
  take1188s versus~40s alone. Do not repeat same-GPU concurrent CFD pilots.
- GPU4 queue `hg-authornorm-offpolicy-20260906` runs TD3 then DDPG authornorm/.5
  short1200 checks, then full100k TD3 using the successful frozen pilot source.
  Current TD3 check is active. No full authornorm DDPG launch queued yet.
  GPU3 remains U_inf TD3-scaled/.5 (~16k); GPU5 U_inf TD3-scaled/.25 (~40k),
  best-207.676 remains much worse than paper. GPU7 Pinball3Dzero1000 (~900).
- NACA73M-cell smoke2 passed onGPU2, wall973s including~800s mesh setup,
  ~62599MiB GPU memory. First forces(.39052,.68388), second(.40512,.72468).
  `hg-naca3d-gust-zero25-20260906` is now active onGPU2 (again expensive init).
  The actual218500 restart and public200000+18500 gust-generation stage align;
  do not blindly use properties_run_gust.toml with the later restart. See
  `naca-source-restart-audit-20260906.json`. Complete gust/control window is
  unvalidated; no NACA policy trained.
- JAX full-policy2-control gradient check passed a perturbation-convergence
  study: norm83.90117, FD error .633% at1e-6 and .0063% at1e-7. Older1e-5 check
  remains failed. Full10-control check has huge gradient1.7552e13 and failed
  ordinary-vs-valuegrad forward parity (TKE24.219045 vs24.215397), before FD.
  This long chaotic trajectory is NOT a validated GPPO gradient or baseline.
  Run `hg-jax-full-policy-gradient10-20260906` exited nonzero, no trained artifact.
- Original GPU JAX5000 is still active onGPU0 (~1520); CPU5000 remains on
  cores192–207. New optimizer-side constant reward scale.01 pilot80 is active
  on CPU160–175 (`hg-jax-ppo-scaled-check-20260906`). Physical objective and
  all reported/evaluated returns remain UNSCALED; scale is separately recorded.
  Persistent queue `hg-jax-scaled-full-20260906` waits for that pilot AND the
  original GPU5000 to finish0, then runs scaledGPU5000 onGPU0 from frozen source.
  This is a documented normalization ablation, not original final-paper config.
- Author34 Cavity16000 calibration is at~14600 onGPU6. Persistent queue
  `hg-cavity-calibrated-20260906` waits for completion, uses in-image R4 scripts
  to freeze calibration and actual final restart, then runs PPO/DDPG/TD3 each
  1200 with fresh replay at .02 before starting full100k PPO onGPU6. It must
  pass cycle/stationarity checks; do not precreate its exclusive output files:
  `cavity-calibration-author34-20260906.json`,
  `cavity-restart-author34-16000-20260906.json`.
- `hf models list --search hydrogym` returned no public model repositories.
  Public CFD `.ckpt` files are solver states, not learned controller weights.
  No paper scores are accepted, release suite remains unvalidated. No authors,
  agents, external uploads, commits or pushes were used.

## Active work — approximately 13:03 UTC

- Author34 cavity calibration completed:16,000 controls,1,203 estimated tail
  cycles/peaks, max block-mean drift0.0012768std. Reconstruction vectors are
  `cavity-calibration-author34-20260906.json`; actual restart step6,925,002 SHA
  `b0da440a2c39f82a5ef7adfdc0972fcb6f8994c328757d5ad3ebc5d27d17f843`.
  These are reconstructed, not original paper initial conditions/targets.
  Zero100 return-6802.6686167. PPO/DDPG/TD3 each completed1200 controls and
  fresh final/best replays using in-image R4 code and physical bound.02.
  Full100k PPO started12:59:45 onGPU6 in
  `hg-cavity2d-ppo-calibrated-s42-20260906`, frozen pilot source.
- Author-normalization TD3/DDPG Pinball1200 checks completed; full100k
  `hg-pinball2d-td3-authornorm-s42-20260906` is active onGPU4. Full PPO200k
  remains onGPU1 (~11k). Earlier U_inf TD3/.5 (GPU3) and /.25 (GPU5) continue.
- 3D Pinball1000-control zero diagnostic completed: CD2.7284905, last-half
  CD2.7459395, RMSCL0.0570980, wall13514.5seconds. The physical gap to the
  paper table is not removed by a longer window. Preserve it as a diagnostic.
  New public-example175-probe y,x,z ordering passed a CPU coordinate test.
  `hg-pinball3d-ppo-public-check-20260906` is active onGPU7 (PPO400, fresh
  final/best replays); tmux `hg-pinball3d-public-20260906` starts100k only after
  successful completion. See PINBALL3D_RECONSTRUCTION.md; no3D learned score yet.
- NACAzero25 completed: return-8.142697, meanCL1.093214, RMSCL1.114852,
  wall2597.9seconds including initialization. It has not reproduced the full
  Fig.2c gust peak. The public short horizon is not certified figure metadata;
  `hg-naca3d-gust-zero50-20260906` now extends an explicitly labelled continuous
  diagnostic onGPU2. Do not silently replace the scored horizon or claim a
  gust-free setup; the verified restart follows the author's gust-generation stage.
- GPU JAX5000 remains active; CPU5000 remains on192–207. Scale.01 CPU80
  pilot completed with160 updates, parameter delta0.57705 and unscaled negative
  returns. The already-queued second GPU5000 is a normalization ablation,
  not an independent seed or recovered original experiment.
- Full-policy two-control gradients passed, ten-control gradient failed. A
  separate **Supplement Eq.34 alternative** is implemented: optimizer reward
  r+(dr/da).a over each full control interval. Both real FP64 interval checks pass
  forward parity and physical TKE finite differences (156.5s). This is NOT the
  undisclosed adaptive Eq.32 GPPO. `jax_public.py` labels it
  PPO-Eq34-reconstruction and reports only unaugmented physical returns.
  `hg-jax-eq34-ppo-check-20260906` is active onCPU160–175,40 controls; no training
  success claim until actual updates/checkpoint and raw-return audit finish.
- JAX actor-only Flax export passes64 input comparisons, max error0. Same-CPU
  native vs resident-JSONL replays have identical complete10-control trajectories
  and return12.735181093215942. `jax-resident-parity-20260906.json` records model
  and package hashes. The earlier GPU replay13.623970 is a different backend,
  not export parity evidence. No paper performance is accepted from either.
- R4 remains the last built image. New175-probe, Eq.34 and standalone JAX
  export/replay code is currently host/frozen-source; rebuild and test the actual
  image before declaring it shipped. No validated full suite or accepted anchors.

## Active work — approximately 13:47 UTC

- R5 image built and tested: `0e7c595fb5ea1ee230ef47ef68f57b4643514d2dd80348ea284c21d1dc245212`.
  Actual in-image task25 tests, cavity/probes6, cavity learners3 and Pinball
  recipe9 checks passed, followed by a trained JAX actor export and resident
  JSON-lines handshake. Receipt `r5-validation-20260906/completed.json`.
  New rollout40/time-limit changes below postdate R5 and need another build.
- Confirmed a manual MAIA adaptation error: setting native max_episode_steps=199
  to obtain200 controls marks the final transition terminated. Public author
  TorchRL StepCounter instead truncates at200 before native `iter > limit`,
  preserving critic bootstrap. New opt-in `--bootstrap-time-limits` matches
  all200 StepCounter transitions and refuses unexpected native completion.
  Protocol/config provenance and independent replay checks distinguish legacy
  and corrected runs. See TIME_LIMIT_AUDIT.md. This is not yet a demonstrated
  explanation of the paper performance gap.
- `hg-time-limit-learner-tests-v2-20260906` exited0: six boundary/GAE tests,
  all9 Pinball recipe real synthetic update/reloads and three cavity learners.
  First launch used a wrong runtime path and exited127; preserved, not a pass.
  Actual corrected zero200 `hg-pinball2d-bootstrap-zero-20260906` reproduces
  legacy return-587.8809887800147 and drag2.9054702151084175 exactly, now with
  terminatedFalse/truncatedTrue. Physics/reward are unchanged.
- Stopped and preserved legacy author-normalized PPO200k at24k checkpoint
  (candidate5704ba90..., return-1069.1385), and U_inf/.5 TD3 at40k checkpoint
  (candidatec628ebdc..., return-392.3311). Audits are ppo-bootstrap-reallocation
  and bootstrap-reallocation JSONs. They remain partial, not full-budget seeds.
  GPU3 is running corrected PPO1200 then DDPG1200/TD31200 plus independent
  reloads, followed by TD3-scaled/.25 100k. GPU1 waits for corrected PPO pilot
  then reruns the identical author recipe/normalization/seed200k with bootstrap.
  Original 3D/Cylinder tmux queues were paused after verifying they only waited
  on containers. New Cylinder queue uses corrected semantics after GPU5 frees.
  Legacy 3D400 pilot remains live onGPU7; review before spending full100k.
- Eq34 CPU40 pilot completed: four physical episodes,80 updates, parameter
  delta.29329556; checkpoint7bfa9547.... Raw-return audit confirms reported
  scores are physical rewards, not augmented optimizer rewards. FullGPU5000
  alternative is queued after ordinary and reward-scaled PPO5000; not Eq32 GPPO.
- Actual public rollout40 CPU160 pilot passed:16episodes,80updates, parameter
  delta.34244075, checkpointb8c38ae3..., wall1030s. Earlier full runs use
  rollout10/minibatch5; public source uses40/minibatch20. Keep distinct.
- Public cited GI-PPO source located and cloned read-only for inspection:
  SonSang/gippo revisiond7b741876d41d7e0a35556c303217c92cabc3a86. It uses an
  alpha-policy/action-gradient update, not evidence of HydroGym's exact GPPO
  weighting. No author contact, no claim that this is the released Fig.3 trainer.
- No accepted full-suite anchors, no certified paper-performance reproduction.
  No agents, commits, pushes or uploads. Continue measured baseline work.

## Active work — approximately 13:54 UTC

- R6 built and actually tested, SHA
  `4074fe394f8c3fe0c284a5f37a36e484fe224a3c10c8c67d7145389b79b39990`.
  Receipt `r6-validation-20260906/completed.json`: task25, boundary/probe/cavity/
  grouping13 tests, all3 cavity learners and9 Pinball recipes, real trained
  actor-only export and resident inference. R6 includes rollout40 and corrected
  native time limits. Later host README warnings and convergence test postdate it.
- Corrected Pinball PPO1200 plus fresh final/best evaluations completed0.
  GPU1 now runs `hg-pinball2d-ppo-authornorm-bootstrap-200k-s42-20260906`;
  GPU3 runs DDPG1200 then TD31200 before corrected TD3-scaled/.25 100k.
  GPU7 now runs corrected `hg-pinball3d-ppo-bootstrap-check-20260906`;
  its old pilot was stopped/preserved at initial policy, audit
  `three-dimensional-bootstrap-reallocation-20260906.json`. Corrected400
  must pass actual training/reloads before the queue starts100k.
- Legacy cavity PPO stopped/preserved at8k checkpoint, audit
  `cavity-bootstrap-reallocation-20260906.json`. New R6 corrected zero100
  return-6802.668616663734 matches the old physical return. GPU6 now runs
  corrected PPO1200, then DDPG/TD3 checks and full PPO100k, all in-image R6.
  `bootstrap-native-parity-20260906.json` records exact Pinball shared-field
  trajectory parity. Legacy logs lacked observations/per-body coefficients;
  no claim is made about equality of absent fields.
- JAX firstGPU5000 completed500episodes,10,000updates, parameterL2delta5.69875,
  wall8120.08sec. Final checkpoint446ec468b77d327012b223ce764e9f28e01aa7393b08dc5ef47db27d626ed1ae.
  FreshCPU replay and actor export passed; final policy chooses zero throughout,
  return29.7304080725, meanTKE2.9730408. This has not established mixing improvement
  or full paper/setup parity. Scaled5000 now runs alone onGPU0; Eq34 full queued.
- New GPU2 queue waits NACAzero50, then performs same-GPU direct/zero/resident
  replays of the complete5000 checkpoint. Afterwards it runs public-rollout40
  unscaledPPO5120controls (512episodes, explicit2.4% budget difference), using
  passedCPU40 pilot and actualR6 code. Do not share GPU2 before NACA exits.
- `hg-learner-convergence-20260906` runs a CPU-only known-optimum control test
  of actualR6 learners,20kcontrols each. PPO squared error fell to.001343 of
  initial (return-54.038→-.07258); DDPG/TD3 still running. This checks actual
  optimization, not CFD performance. Test source is postR6, implementations inR6.
- Shared starter still requires time-limit alignment and non-MAIA adapter/
  suite integration before release. Host README explicitly warns about legacy
199 semantics. No suite or paper-performance anchors are validated.

## Active work — approximately 14:07 UTC

- All three actualR6 learner known-optimum20k checks passed. Final/initial
  squared-error ratios: PPO.00134308, DDPG.01304597, TD3.00409309.
  `hg-learner-convergence-20260906/convergence.json` is synthetic convergence
  evidence only, not CFD performance. Test source remains postR6.
- Corrected Pinball DDPG/.5/author-norm1200 completed400 optimizer updates but
  its first trained evaluation diverged at control119. Logged drag progressed
  to1.11e105/1.28e222 at117/118; native finite check then abortedMPI promptly.
  Preserve invalid candidate and traces, no accepted score. The GPU3 chained
  pilot queue stopped (not still pending). TD3 corrected1200 was launched
  independently onGPU3, with a new full100k queue conditional on its success.
- Independent frozen DDPG replay onGPU4 reproduced all118 logged transitions
  exactly, including observations/actions, then the same nonfinite failure.
  It is a reproducible physical instability, not a shutdown/transport failure.
  `ddpg-bootstrap-cfd-exclusion-20260906.json` records exclusion. Host native.py
  now adds failed-transition context and checks float32 policy observations;
  those extra diagnostics postdateR6 and existing frozen runs are unchanged.
- Stopped/preserved legacy TD3/author-norm after20k checkpoint (89ba9a94...,
  return-1416.8181), audittd3-authornorm-bootstrap-reallocation. Only the stronger
  legacyTD3/.25 (GPU5,~68k) remains as the full-budget terminal-semantics control.
  GPU4 now runs corrected DDPG/.25/U_inf1200, then full100k if native replay passes.
  This changes both norm/bound relative to the failed.5 trial, so it cannot
  establish actuator-bound causality alone; it is a separate source-bound recipe.
- Portable cavity inputs were constructed and verified:
  `data/reference-calibrations/Cavity_2D_Re4200-author34-v1` contains actual
  restart, targetvectors, rawmeasurement/provenance and hashes, not paper scores.
  New package_cavity_calibration.py recomputes statistics/header/hashes before
  copying. `hg-cavity-data-only-zero-20260906` used in-imageR6 code, only/dataRO
  and a fresh/output mount (nohistorical/runs), and exactly reproduced100-control
  return-6802.668616663734. Host README now documents portable training inputs.
- Corrected cavity PPO1200+freshreplays passed, finalreturn-21999.1581482
  (not paper-level, but distinct from legacy calibration run). GPU6 continues
  DDPG thenTD3 short checks before fullPPO100k. All use frozenR6 source.
- NACAzero50 still running. Sampled native restart variables are finite and
  nonuniform: u median.11277, min-.10760,max.20663; v min-.11123,max.13120;
  w varies up to.08. This is not proof of a paper-equivalent gust, but excludes
  a trivially uniform stored field. No new NACA training protocol inferred.
- No certified paperperformance or validated fullsuite; no author contact,
  agents, uploads, commits or pushes. Continue goal.

## Active work — approximately 14:43 UTC

- Corrected cavity TD3 1200 and fresh final/best replays passed. Full corrected
  cavity PPO100k now runs on GPU6. Corrected Pinball DDPG/.25/U_inf100k and
  TD3-scaled/.25/100k are running on GPUs4/3 after their successful pilots.
  The 3D Pinball initial200 evaluation completed with return-558.637805945771;
  actual400-control training is underway, not yet a trained3D result.
- NACAzero50 completed, peakCL1.5447344 still not paper~4.4. Its first25
  transitions agree numerically, not bitwise, with the earlier25 test.
  NACA_RECONSTRUCTION.md records actual maxima of force/observation differences.
- FullGPU5000 PPO checkpoint direct/resident replays have identical actions,
  rewards and TKE, return27.9438669682, but observation maxdifference4.76837e-7.
  Earlier whole-trajectory exactness wording was too strong. New host audit
  `jax-5000-gpu2-parity-audit-20260906.json` preserves the actual differences.
  Zero-control behavior and different train/eval aggregation prevent a full
  paper-performance claim despite a numerically close PPO~28 endpoint.
- R7 rebuilt after fixing the mock test's lazy MPI import. Final image SHA
  `824edb1fc8c6939391bd44e93ac2dd3b9315d0f7af1b21facb9353965c299250`.
  `hg-r7-image-tests-20260906` now tests actual image implementations including
  three float32 observation guards and re-materializing the portable cavity
  calibration bundle. It is not yet a passed receipt. Ongoing scientific runs
  retain their frozen earlier source; no hot updates to their code.

- R7 test's first container exited126 because the manual command omitted `bash`
  for a non-executable runtime script; no tests ran. Documented task entrypoints
  already use `bash`. Corrected container `hg-r7-image-tests-v2-20260906`
  completed0:25task tests,16baseline tests,3cavity learners,9Pinball recipes,
  actor-only export/resident inference, and exact cavity bundle payload hashes.
  Receipt `r7-validation-20260906/completed.json`. No GPU was used by these tests.
- Host shared MAIA path now has explicit `episode_end=time_limit` (native limit
  equals horizon) through a common training/replay wrapper, with legacy default
  `native` unchanged. Mixed conventions/early native completion/post-timeout
  stepping are rejected; shared resume rejects a semantics switch. Host CPU29
  tests passed with4 ML skips, so full image tests and actual native verification
  are still required. These source changes postdate R7 and live frozen runs.

## Active work — approximately 14:55 UTC

- `hg-shared-bootstrap-cpu-check-20260906` exited0 with all29 task tests, no
  skips. Host source mounted into R7 runtime. Actual SB3 rollout bootstrap
  verified: reward-1/value5/gamma.99 gives finalbufferreward3.95 at truncation.
- R8 build is still completing (PTY36266). Its context includes shared runtime
  boundary corrections but predates the new optional flag in
  baselines/shared_native_check.py. Therefore the forthcoming native check uses
  frozen `/mnt/workspace/hydrogym/reproduction/shared-time-limit-native-check-20260906.py`
  mounted as /diagnostic.py, while training/evaluator implementations come from
  the R8 image. Do not claim this launcher update was copied into R8.
- New tmux `hg-shared-time-limit-native-20260906` waits for the GPU2 JAX public
  rollout40/5120 run to finish, requires the R8 passed image-test receipt, then
  runs shared Pinball400 + Cylinder400 + two relocated200-control evaluations.
  No GPU sharing; no accepted suite/anchors fabricated for this diagnostic.
- NACA audit now records the reward-sign inconsistency: Fig2c positive scores
  versus article/Eq37/source nonpositive raw reward. No arbitrary plot offset.
- Current scientific runs continue: scaledJAX~2400/5000, publicrolloutJAX1440/5120,
  CPUJAX~2400/5000; corrected PinballPPO~17k/200k, TD3/DDPG~10k/100k,
  legacyTD3~78k/100k, corrected cavityPPO~6k/100k, corrected3D training underway.
  Early corrected scores remain poor; no unsupported claim that time limits
  alone fix performance. No source edits to running experiments.

## R8 validation — approximately 15:00 UTC

- R8 built SHA `9cf8aaff497460fc6d4c9a7a481698a4cce10e85d653c8552f7d5646ca13e533`.
  `hg-r8-image-tests-20260906` exited0. Actual image implementations passed29
  task tests (no skips, including actual SB3 timeout bootstrap),16manual-baseline
  unit tests,3cavity learner checks,9Pinball recipe checks, portable cavity-data
  reconstruction with identical payload hashes, and trained JAX actor-only
  export/resident inference. Receipt `r8-validation-20260906/completed.json`
  also hashes both shared runtime code and manual baseline code and confirms
  editable/production evaluator source copies are identical.
- The queued native shared check may now proceed when GPU2's current JAX job
  completes. No trained CFD/shared-score result from R8 is claimed yet.
- Further source audit records that older related JAX code uses incoming-state
  reward/observations and8 probes, versus the current new-state reward/64probes.
  Both integrators advance9990 rather than10000steps per nominal10-unit action
  due to floating floor division. Existing experiments retain that source
  behavior; JAX_RECONSTRUCTION.md distinguishes nominal from integrated time.
- No final response, accepted full suite, author contact, agent launches,
  commits, pushes or uploads. Paper-performance goal remains active.

## Monitoring / comparison preparation — approximately 15:15 UTC

- Native corrected DDPG and TD3 are now around16k/100k. At12k, deterministic
  mean actions were approximately[.121,-.883,.853] and[.092,-.717,.740], respectively:
  learned counter-rotation is present, but RMSlift~1.4 keeps returns around-460.
  PPO/author-norm20k return-1133.8533 remains worse than its initialization.
  Its measured observations stay modest (absolute max<5 through16k), so no
  evidence of gross preprocessing overflow explains that current gap.
- 3D Pinball's Python/MPI and native solver processes remain active (~2CPU
  cores total), initial200 evaluation complete and training underway. No trained
  checkpoint yet. CavityPPO~9k, legacyTD3~82k, scaledJAX~3240/5000 and
  publicrolloutJAX~2080/5120 continue. Do not stop early because scores are low.
- New tmux `hg-jax-final-gpu0-comparison-20260906` waits scaled5000, queuedEq34
  5000, and publicrollout40/5120 completion, then exports all4 JAX candidates
  afresh onCPU and evaluates original/resident pairs sequentially onGPU0 with
  R8. Includes firstfullPPO for a same-device comparison; it does not pool the
  previously different CPU/GPU trajectories. Source runs are read-only mounts.
  Queue script and log are under reproduction/queue-jax-final-gpu0-comparison-20260906.*.
- A further public-source check did not recover missing3D cavity assets.
  Supplement geometry/refinement prose and existing2D resource properties do
  not uniquely specify the original3D mesh/restart. No invented3D asset created.
- All builds/PTYS have finished. Latest tested image is R8, not R7/R6. No current
  user-facing final; active goal still requires paper-level results and release.

## Measured progress — approximately 15:21 UTC

- 3D Pinball first actual training200 completed: return-903.2228181795571,
  meanCD3.29385245, RMSCL1.5032413, wall2549.35sec (42.5min). It has started the
  second200-control episode; the first400-transition PPO update is still ahead.
  Native trajectory finite; slow progress is measured, not a presumed hang.
- Corrected DDPG16k native evaluation improved to-240.39065816773865,
  meanCD.6340424385, RMSCL.6662371307, fixed last-halfCD.3652610938.
  Not a full-budget or paper-level result. Corrected TD3 at16k returned-528.09845;
  do not conflate algorithms or select only favorable checkpoints as finals.
- `hg-ddpg-bootstrap16k-export-20260906` exited0 in R8, CPU-only. Immutable
  source policy-0016000.pt SHA
  `ff4a05652aab4db742594b813e645b3bf908fa41e0b08b748310793192a0a595`,
  exported actor SHA
  `c5ed9f2fbb65e2b775775d867e1eba052f7c4edc3e8d9fd54fb2ff56b3d9433d`.
  Export64inputs and all200 recorded native observations via resident protocol
  had maxactionerror5.960464477539063e-8. Receipt in that run/completed.json.
  This is recorded-observation parity, not fresh closed-loop CFD evidence.
  It remains under reproduction/, not a certified data score anchor or shared
  task submission. Source run mounted read-only; current training unchanged.
- ScaledJAX3520/5000, publicrolloutJAX2400/5120, legacyTD3~84k/100k;
  other corrected runs continue. R8 remains latest tested image; allPTYSclosed.

## Evidence packaging / monitoring — approximately 15:48 UTC

- The host archiver now accepts independently replayed Cavity workflows and
  retains all small frozen source files, including recipes and launch scripts,
  rather than only Python files. It skips bytecode caches and rejects source
  links escaping the snapshot. Archive and summary unit tests passed.
- Three corrected Cavity1200 pilots were archived successfully (90 files) at
  `reproduction/cavity-bootstrap-pilot-archive-20260906`. They remain pilots,
  not data score anchors or paper-matched baselines. These archiver edits and
  tests postdate the R8 image; no claim that they are already inside R8.
- At15:47, scaledGPUJAX4480/5000, publicrollout40JAX3360/5120 and CPUJAX2960/5000
  were active. Corrected Pinball latest evaluations: PPO28k return-1035.78367,
  DDPG24k-456.72567 (best16k-240.39066), TD3scaled20k-382.45642.
  Corrected Cavity16k returned-133649.82490, below its initial policy.
  LegacyTD3 reached88k/100k, latest return-302.12440, best20k-207.67639.
- All8 GPUs have one live scientific training job each; native3D remains in
  its first400-transition collection. No author contact, external upload,
  commits/pushes, source changes to live experiments or acceptance promotion.

## Completed scaled JAX / first trained3D artifact — approximately16:01 UTC

- `hg-jax-gpu-ppo-scaled5000-20260906` exited0 after5000 controls/500 episodes,
  10000 optimizer updates,7664.1857sec. ParameterdeltaL2=3.50558424;
  checkpointSHA20896153c8f511e9224607afaaa59d1669eb490f39defb25c9479f6a5bd5f131.
  Independent CPU10 replay and actor export both exited0. Return29.73040807247,
  all10 actions zero, entire logged trajectory identical to the earlier
  unscaled5000 candidate's same-CPU replay. Export64inputs maxerror0;
  actorSHA85d7e93b76daea3d1797b4a84fec59ad0a2c13486d33ae694461ba2ed677cb05.
  Audit `reproduction/jax-scaled5000-cpu-comparison-20260906.json`.
- The queued full Eq34 reconstruction started on GPU0 after scaled exit0,
  `hg-jax-gpu-eq34-5000-20260906`, using the passed CPU40 pilot's frozen source
  in R4. Compilation/startup in progress; no gradient-enhanced training result
  claimed yet. Eq34 remains distinct from the original adaptive Eq32 GPPO.
- 3D Pinball400 collection completed: second200 return-862.17574853,
  wall2493.054sec.400optimizerupdates completed, finite loss; immutable
  policy-0000400.pt independently inspected: actorweightdeltaL2=1.80910894,
  allactorweightsfinite, time_limit_bootstrapTrue. Updated-policy native
  evaluation is underway; full pilot workflow is not complete.
- Publicrollout40JAX reached3840/5120. Other native full budgets continue.
  Tests archive/summary passed again after documentation update. No full-suite
  acceptance, final response, external contact, agent launch, push or upload.

## Cavity optimization comparison queued — approximately16:14 UTC

- Current corrected Cavity PPO remains poor through roughly20k controls,
  with large critic losses; it is not stopped. Added an explicitly separate
  `ppo-scaled` comparison: optimizer reward x.1, criticLR.001 instead of.0003,
  same seed42/physical settings/targets/restart/evaluation units. Two changed
  optimizer settings; no isolated-cause or paper-equivalence claim.
- Extended test_cavity_learner.py to four cases (including this recipe), with
  bootstrap=True and explicit reward-scale assertions. Frozen test mounted into
  R8; actual in-image learners passed all4 cases, `hg-cavity-scaled-learner-check-20260906`
  exit0. This test edit postdates R8; no image rebuilt and no live source changed.
- New tmux `hg-cavity-scaled-gpu2-20260906`, script/log
  `reproduction/queue-cavity-scaled-gpu2-20260906.*`, waits CPUtestexit0,
  publicrollout40JAXexit0, and shared-time-limit-nativeexit0, plus R8 receipt.
  Then sequential GPU2 runs `hg-cavity2d-ppo-scaled-bootstrap-check-20260906`
  (1200 controls, eval400, fresh final/best), then
  `hg-cavity2d-ppo-scaled-bootstrap-s42-20260906` (100000, eval4000).
  Both use data-only portable calibration/restart inputs and a separate
  output mount; result roots are each `reproduction/RUN_NAME/run`.
- Eq34GPU progressed120/5000, finite gradient/CFD/learner checks. The first
  GPU40 rollout's physical and augmented returns are logged separately;
  its gradient-based training reward is not used as the benchmark score.
- LegacyTD3 completed96k evaluation(-468.56463), still training to100k.
  No final response or accepted full suite; goal remains active.

## Public-rollout PPO complete / shared native running — approximately16:32 UTC

- Publicrollout40 GPU PPO5120 exited0,512actualepisodes/2560optimizerupdates,
  parameterdeltaL2=3.12839293, wall8015.74564sec. CheckpointSHA
  d42c5dc5a14fc7ca7d3059f6f634f86aa42e46e522795a12af9cae3e17bba23b.
- Added immediate CPU160-175 queue `hg-jax-rollout40-independent-cpu-20260906`
  so its fresh evaluation/export/resident replay need not wait for Eq34GPU.
  Allthree containers exited0 in R8. Direct/resident CPUreturn29.73040807247,
  all10actionszero, entire logged trajectories equal. Export64inputs maxerror0,
  actorSHA33628df23e0a60a403d53d291e37a88aa085889d46be78fa7db464d6535d536e.
  Audit `reproduction/jax-rollout40-5120-cpu-comparison-20260906.json`.
  This is not enhanced mixing, independent-seed replication or GPPO reproduction.
- After its predecessor completed, `hg-shared-time-limit-native-20260906` started
  GPU2 in R8, finished Pinball400 training and began continuing the same
  checkpoint on Cylinder400. Both final relocated200-control replays are pending.
  Cavityscaled queue waits for this complete workflow, with no GPU overlap.
- LegacyTD3/.25 completed100000 training controls and its final training-process
  evaluation: return-270.4765330446075, meanCD.4958680775568696,
  RMSCL.8886920469347198, last-halfCD.4355039384774889. Independent final/best
  workflow subsequently completed at16:32, both replay scores matching.
- Eq34GPU240/5000 at16:26, other corrected native budgets continue, and3D's
  trained400 checkpoint is still in its first200-control evaluation.

## LegacyTD3 archive / Cylinder starting — approximately16:33 UTC

- Full500episode legacyTD3scaled seed42 workflow exited0. Host archiver validated
  training counts, checkpoint hashes and final replay parity and saved32files
  to `reproduction/pinball-td3-scaled100k-archive-20260906` (not certified anchors).
  Full-window final drag reduction82.9333%, last-half85.0119%; best20k
  return-207.67638528525, full-window84.4210%, last-half89.6045%. Large lift
  fluctuations and one seed prevent claiming the paper's three-seed result.
  Its legacy terminal semantics remain explicit, not pooled with corrected runs.
- GPU5 queue moved to `hg-cylinder2d-ppo-bootstrap-check-20260906` after
  legacyworkflowexit0. The400-control pilot plus fresh final/best evaluations
  must pass before its planned500000-control full Cylinder trial.

## 3D cost correction — approximately16:36 UTC

- Actual two trainingepisodes:400controls/5042.40400029sec =>12.60601sec/control.
  Linear100k extrapolation14.5903days, plus.7295day for25x200control evaluations
  and additional overhead. Prior~8day estimates were too optimistic; use this
  measured throughput, not a completion-time promise. User explicitly updated.
- Read-only source audit found upstream documented multi-rank MAIA launches
  (`examples/maia/getting_started/run_example_docker.sh`, NPROC_MAIA GPUs/cores;
  getting_started/README.md -np4 example). No multi-rank performance/parity test
  yet, no current solver-rank/GPU allocation or physics change.

## Shared integration passed / CPU capacity investigation — approximately16:50 UTC

- `hg-shared-time-limit-native-20260906` exited0. One checkpoint trained400
  controls on Pinball then400 on Cylinder, and relocated200-control replays
  returned-528.9032235150/-532.1742954065 with terminatedFalse/truncatedTrue.
  SubmissionSHAa3792cc108fd57f59f400f2898b0669ae2fceccb7e9533b020d4e24d9559bd5c.
  Task readiness/docs now accurately say two-case native integration verified,
  but full-suite/paper baseline release still pending; no suite/anchor promotion.
- GPU2 moved to Cavityscaled1200 pilot using data-only bundle. At800 its
  evaluation returned-26812.36198643; no full or accepted performance result yet.
- Corrected3D's first trained400 native eval completed: return-649.4809060688,
  meanCD2.8886494884,RMSCL.4298236163,wall2467.9396sec. It is replaying final
  and initial-best independently; full pilot workflow is still pending.
- CPU MAIA runtime exists in R8: module1.0-gompi-2024a-SystemCUDA-CPU and
  hydrogym_cpu venv. Existing native.py already supports --solver-processes.
  CPU32/256GiB and CPU8/384GiB20-control diagnostics both hit their own cgroup
  OOM limits during initialization, with no valid flow steps. Per-rank default
  maxNoCells80M causes large allocations; CPU8 loaded2.42M local rank0cells.
  Both failures/container outputs preserved; separate memory-exclusion JSONs
  in reproduction/. No GPU training stopped or physics/source data changed.
- Added optional --max-cells to native.py: only one private runtime-properties
  maxNoCells substitution, no source overwrite, hashes and original/effective
  capacities logged, independent evaluation propagates it. Default unchanged.
 4pure capacity tests plus grouping/archive passed onhost. An attempted broader
  host test invocation lacked numpy/torch; reran in R8 runtime with frozen host
  source: `hg-native-capacity-tests-20260906` exited0, all16 tests, no skips.
- Frozen updated source at reproduction/pinball-cpu-capacity-source-20260906;
  nativeSHA9339de94f1a3a0dab19b27ecb0636a6766921e9b20dcfc1f7c385c616e14d188.
  This and recent tests/archive/docs postdate R8; no R9image built.
  CPU runtime script reproduction/maia-cpu-runtime-20260906.sh SHA
  7a959aa6a25ea08722c2cb9208ffa4c58813417dc4eacff4928275f1c8e5d243.
- New `hg-pinball3d-cpu8-cap20m-zero20-20260906` uses this frozen source in R8
  runtime,CPU32-47,8MAIAranks,384GiB cap,noGPU. Only maxNoCells20M private
  override; same published mesh/restart/physical settings/175probes. Need verify
  actual loaded global cell count, finite trajectory and throughput; no claimed
  mesh/parity/speed result yet. Output root is reproduction/RUN_NAME/run.

## R9 tested / full Cavity and Cylinder started — approximately17:11 UTC

- R9 development image built and its actual copied implementations passed:
  `ai4sci-hydrogym-task:baseline-r9-20260906`, image SHA
  5a8398d47ea2b11f439feb7626ef2a053c7d2fabecf4f6dee7510e0c355cee74.
  `hg-r9-image-tests-20260906` exited0:29 task tests/no skips,21 baseline
  boundary/probe/precision/capacity/archive tests,4 Cavity learner cases,
  9 Pinball recipes, portable calibration payload hashes exact, trained JAX
  actor export and resident protocol. Receipt:reproduction/r9-validation-20260906.
  Native SHA188798993c2049c7325e9ae076b56f41802a23bbdf98efed5290b02d1df84d78
  includes optional evaluation step timing and private allocation-capacity override.
  These checks do not establish paper performance or change running sources.
- Cavityscaled1200 pilot/fresh final/best all completed:final-48323.74634550,
  best400-22254.05242072. Archived30files in
  reproduction/cavity-scaled-bootstrap-pilot-archive-20260906. Full100k started
  GPU2 at~16:53 with R8 and data-only portable calibration/restart.
- Cylinder400 pilot/fresh replay completed:final-499.15465222,
  meanCD1.49350057,RMSCL1.16039016. Archived40files in
  reproduction/cylinder-bootstrap-pilot-archive-20260906. Full500k started
  GPU5 at~16:49;2500episodes covers the figure's approximate endpoint, not a
  recovered exact endpoint. Both full trials are still early and not accepted.
- CPU8/maxNoCells20M successfully loaded original16,465,722 global cells;
  rank0 has2,420,113 local cells. RSS~123.7GiB, noOOM. First force/reward
  agrees with prior GPUzero to~1e-13, but current CPU throughput is much slower.
  Its20-control diagnostic is still running; no speedup or trained result.
- Read-only20sec profiling of ongoing GPU3D fresh replay:Python399samples,
  390 waiting in runTimeSteps and9 in probe receive. Native perf1913samples,
  30.01% in deletePeriodicConnection and19.29% in restorePeriodicConnection.
  This is a bounded CPU-side sample, not GPU kernel profiling or an established
  optimization. Profiles preserved in reproduction/pinball3d-*-20260906.*.
  Existing GPU3D pilot and its full-run queue remain unchanged at this point.

## 3D allocation comparison interposed — approximately17:16 UTC

- R9 optional step-timing mock test also passed in its actual copied native.py
  (`hg-r9-native-timing-unit-20260906` exit0). Field absent by default, .125sec
  under controlled clock, reward unchanged. Separate script in reproduction/.
- Stopped only waiting tmux queue `hg-pinball3d-bootstrap-20260906`, preserving
  script/log, and replaced it with `hg-pinball3d-capacity-full-20260906`.
  Live3D container remained running with the same PID. New queue waits for all
  current400-pilot independent replays, then sequential GPU7 zero20 diagnostics
  at original80M allocation and private20M allocation, using immutable R9 image ID.
  No running experiment, physical setting, mesh or restart was changed.
- The comparison checks20 finite transitions (forces/reward/actions/350-input
  observations) within1e-6, original16,465,722 loaded global cells, completion
  receipts, matching protocols and private properties differing only in capacity.
  Select20M only if mean per-control time after the first is at least10% faster;
  otherwise keep original allocation. Any invalid/mismatched diagnostic stops
  this queue before full launch for investigation. This rule is an allocation
  diagnostic, not selection on benchmark performance.
- Then launch100k publicPPO/seed42 with unchanged physics, R9 native wrapper,
  same byte-identical public_learner.py and time_limits.py as the passed pilot.
  Full result root will be reproduction/hg-pinball3d-ppo-bootstrap-s42-20260906/run.
  No result or acceleration claimed before these checks finish.
- Native compiler metadata is optimized production (-O2, stdpar=gpu,
  cc90/cc100, managed memory), not an unoptimized build. Installed src/ contains
  object/dependency/build metadata, not the original solver source tree.

## 3D trained replay mismatch — approximately17:24 UTC

- Fresh trained400 replay completed200 controls but returned-675.4765636553,
  compared with-649.4809060688 in the training process. Both use checkpoint
  SHA0acb5c81adaeffbd437ebd681d01ffa9015c889ec9b06c991686b8f204b23a81 and
  exactly equal protocol JSON. First transition is identical; first difference
  appears in step2 next_observation (89components,max1.1280179e-4), while its
  action/reward still match. Later closed-loop feedback amplifies differences.
  Full200 maximum differences:normalizedaction.2416973,reward.9474084,
  CD.2890131,CL.8824622. This is NOT a passed independent reproduction.
- Stopped only waiting `hg-pinball3d-capacity-full-20260906`; no full3D training
  or allocation comparison will start from it. Preserved its files. Live3D
  container/PID767387 remains running and is independently replaying initial-best.
  Audit:reproduction/pinball3d-trained-replay-mismatch-20260906.json.
- New waiting GPU7 queue `hg-pinball3d-replay-diagnostic-20260906` uses immutable
  tested R9 after the current workflow exits. Two fresh processes, each two
  resets/20control episodes with fixed normalized actions equal to the trained
  policy's first action:[.02287449874,.07211599499,-.05839646235]. Native physics
  unchanged. These are repeatability diagnostics, not trained performance.
- New CPU-only queue `hg-pinball3d-cpu-controlled-20260906` waits the current
  zero20 diagnostic, then runs the same fixed sequence/two resets on8 CPUranks
  with the already successful20M allocation. Same data and R9 native wrapper.
  Record per-control timings and all force/probe outputs. No GPU overlaps or
  solver edits. Reset state vs numerical nondeterminism remains unresolved.

## CPUzero20 complete / early cross-process difference — approximately17:30 UTC

- CPU8/20M zero20 exited0 at17:29:05 after its finite solver teardown, with
 20controls/return-54.97416568003,meanCD2.69504641,RMSCL.06167954,
  episodewall2212.8962sec. It is slower than singleGPU, not an acceleration.
  Compared with prior GPUzero-long's first20 controls:return-54.97305734922;
  maximum absolute differences reward.00241237,CD.000105652,CL.00230672.
  Differences exceed1e-6 atstep2. Firststep agreement alone was insufficient;
  probe sets differ175vs192, so this comparison is force-only, not obs parity.
- The queued CPU8 controlled two-reset diagnostic has now started after exit0,
  with no overlapping CPUsolver allocations. GPU7's initial-best replay remains
  running; even its first8 controls differ from the original initial-policy
  episode starting atstep2 nextobs(max~1.0249e-4), although both are fresh
  processes. This argues against blaming training updates alone, but does not
  establish a cause or acceptable statistical evaluation variance.
- Corrected2D TD3 reached44k and improved its native evaluation to-235.67571;
  Cavityscaled4k returned-68492.74439, below initial. Eq34GPU800/5000,
  CPUoriginalJAX~3840/5000. Full budgets continue; no accepted baseline yet.

## Probe-read diagnostic queued — approximately17:36 UTC

- CPU8 fixed-control test's first2 next-observation vectors are float32-exact
  with the training-process GPU400 evaluation. CPUstep2 versus freshGPU400
  nextobs differs in89components/max1.1280179e-4, despite first actions identical
  and reward agreeing to2.45e-13. Later fixed actions differ from the learned
  policy, so this cross-policy comparison is deliberately limited to the first
  completed common-action interval, not a full-trajectory equivalence claim.
- Added `reproduction/pinball3d-probe-requery-20260906.py`: within the worker,
  intercept each getProbeData call, read twice without advancing CFD, log both
  hashes/maxdifference, and return the original first read. Extra reads are a
  diagnostic, not a training change or an assertion of read-only solver behavior.
  Mock test `hg-pinball3d-probe-hook-unit-20260906` exited0 in R9, checking stable
  and stateful queries, first-read preservation and instrumented MPMD bootstrap.
- Replaced only waiting GPUdiagnostic tmux with
  `hg-pinball3d-replay-diagnostic-v2-20260906`. After the ongoing initial-best
  replay, first run4 fixed-action controls with double queries, then execute the
  already-designed two fresh/two-reset20-control comparisons. Live GPU and CPU
  diagnostic processes remain unchanged. No full3D training is queued.
- Public solver source access is not available anonymously: pinned HydroGym
  docs/docs/installation/maia.md explicitly requires AIA registration; GitLab
  and solver-doc opens returned access-denied/login. No contact/access request,
  authentication attempt or private-source acquisition was made.

## DDPG52k candidate / independent seed extensions — approximately17:41 UTC

- Corrected DDPG52k native evaluation:return-142.58656404,meanCD.4375881343,
  RMSCL.4373299324,last-halfCD.2506641849,last-halfRMSCL.1365912400.
  Full-window drag reduction84.9392%, last-half91.3733%. This is one interim
  seed/window, not paper reproduction or the final three-seed statistic.
- Immutable52k checkpointSHA8bf38ba80955065de5abd3f47a4a5f0e6fd48b422b07aedba58b67446c49fe2f
  exported in R9; `hg-ddpg-bootstrap52k-export-20260906` exited0. ActorSHA
  b3500cbe7bfcf1119631d7d67306002bb8d4647c2860daac53382ce883ad4be6.
  Resident actions on all200 recorded native observations maxerror5.96e-8.
  This is artifact/action parity, not a new closed-loop CFD evaluation.
- Added two waiting sequential seed queues: DDPG GPU4 and TD3scaled GPU3,
  each waits its current corrected seed42 full100k/fresh final+best workflow,
  validates counts/hashes/fresh-final metrics with a frozen archiver, then runs
  independent seeds43 and44 at100k each. Launcher/source/recipe/physics/eval
  interval match the corresponding seed42; only training seed changes. No
  quality threshold, best-seed selection, overlaps or AutoResearch agents.
- Queue script:reproduction/queue-pinball-corrected-seeds-20260906.sh;
  tmux hg-pinball-{ddpg,td3}-corrected-seeds-20260906. Archiver snapshot SHA
  a7820c83234c4710892b8f41f154fe23a8c54fa3aecd62b14c4afa6568d421d0.
  The TD3scaled family remains a documented optimizer ablation, not silently
  grouped with public-recipe or legacy-terminal runs. No seed43/44 launched yet.

## Complete-trajectory comparison utility — approximately17:49 UTC

- Added compare_trajectories.py and four unit tests to baseline starter sources
  (postdate R9; not claiming they are in the tested R9 image). Complete horizon,
  ordered steps, finite numeric values and identical array shapes required;
  missing fields require an explicit justified subset. Reports first divergence,
  maximum/RMS errors and raw returns, but no automatic acceptable CFD tolerance.
  Four new tests plus archive/grouping tests passed on host,6total.
- Applied to trained3D400's complete native/fresh episodes, saving
  reproduction/pinball3d-trained-replay-comparison-detailed-20260906.json.
  It confirms the already-recorded mismatch; no changed acceptance threshold.
  The input hashes bind both full logged files to the comparison.
- CorrectedTD3scaled48k evaluation improved to-222.00279783. Other full runs
  and queued diagnostics continue; no accepted paper baseline, upload or push.

## 3D pilot process complete, not replay-accepted — approximately18:02 UTC

- Original400-control3D training/fresh-final/fresh-initial-best processes all
  exited0. Its workflow-completed.json records execution, not parity acceptance.
  Freshinitial-best return-558.6412371094 versus original-558.6378059458;
  absolute difference.00343116, much smaller than the trained policy's~25.996 gap.
  Complete initial-policy comparison:maxrewarderror.000377790,
  maxnextobservationerror.00196466,maxactionerror.0000623800. Not exact equality.
  Audit:reproduction/pinball3d-initial-replay-comparison-detailed-20260906.json.
- New4-control double-probe diagnostic started GPU7 as queued:
  hg-pinball3d-gpu7-probe-requery4-20260906. It precedes two fresh/two-reset
  fixed-action20 comparisons. CPU8 fixed-action comparison remains active.
  Full3D training still not queued or accepted.
- DDPG56k further improved to-130.15575512,meanCD.43195457,
  last-halfCD.24938878,last-halfRMSCL.03133190. The current.25 actuator bound
  is smaller than the author's.5 example; prior fixed-counterrotation evidence
  and separate.5 pilots are documented already, not newly recovered metadata.
  Preserve this distinction when interpreting proximity to a paper target.

## Double-query4 complete / fixed repeats running — approximately18:07 UTC

- `hg-pinball3d-gpu7-probe-requery4-20260906` completed and exited0. All5
  probe pairs (reset+4 steps), each875 raw values, were exactly equal between
  immediate reads at one CFD state. No stateful repeated-read effect found in
  this bounded test. Raw return-11.11586271238; startup is included in its
  286.3838sec episode timer, so do not infer per-step CFD cost from that total.
- Compare only the first4 common fixed-action controls with the CPU8 run:
  maxrewarddifference2.45e-13,maxCD3.89e-13,maxCL1.44e-13;
  nextobs first differs atstep3,max8.58e-6. This is a four-control prefix
  comparison, not parity of the CPU20 episode or evidence that extra querying
  has no effect on all future dynamics. Fixed actions are exactly equal.
- GPU7 now runs `hg-pinball3d-gpu7-fixed20-repeat1-20260906`, without double
  queries, two20-control episodes; another fresh process follows. CPU8 first
  fixed20 episode finished:return-58.3251897593,meanCD2.86313038,
  RMSCL.06728803,wall2207.5641sec. Its second reset/episode is underway.
- Cavityscaled8k evaluation-48099.49391, still worse than initial;
  Eq34GPU~1200/5000. All continuing full-budget outputs remain non-accepted.

## Fixed-control reset discrepancy confirmed — approximately18:21 UTC

- GPU fixed20 repeat1 completed both episodes: raw returns -58.32662165193
  and -58.34666196071. At the first step after reset, next observations are
  exactly equal, but rewards change -2.716936061788 to -2.766100928056.
  CPU8 independently shows the same first-step discrepancy (within 1e-13),
  with identical first next-observations. Its second episode is still running.
  This is not explained by simply calling the difference floating-point noise.
- The second fresh GPU fixed20 process is running. Full-horizon within-process
  comparison is recorded in pinball3d-gpu-fixed20-reset-comparison-20260906.json;
  cross-process and CPU full comparisons await completed data, not a prefix.
- Pinned 3D properties enable lbUseControlRamping with lbRampPercentage=.95.
  Public reset sends reinit and a single zero-control command. A retained
  actuator transition state is a hypothesis, not an established solver cause.
- Added isolated diagnostic pinball3d-reset-doublezero-20260906.py to duplicate
  only that zero reset command. It adds no CFD steps/warmup and leaves original
  solver/assets and training code unchanged. One mock test passed in R9,
  checking reset-only duplication, exact return/argument forwarding, normal
  action preservation and instrumented MPMD bootstrap. Waiting tmux
  hg-pinball3d-doublezero-20260906 runs two20-control episodes after fresh repeat2;
  no full3D training is queued and this is not yet a fix.
- DDPG60k evaluation worsened to -370.72825910; best remains56k -130.15575512.
  TD3scaled56k improved to -218.49244874. PPOauthornorm68k is -969.86444907.
  Keep complete budgets and all seeds, not a selected favorable checkpoint.

## Evidence validation hardening — approximately18:22 UTC

- Host archive_results.py now explicitly rejects nonnumeric/nonfinite final
  metrics before comparing replays; NaN must not bypass the unchanged1e-5
  comparison. Added invalid-metric and finite-tolerance unit tests. Archive,
  trajectory comparison and grouping tests passed,8 total. This postdates R9
  and does not change already frozen running/queued sources or their outputs.
- The old author 3D properties use control ramping=false, versus current
  pinned assets true/.95. They also differ in domain, solver method, restart
  and boundary IDs. This is documented provenance, not grounds for blindly
  swapping incompatible files or declaring ramping the reset discrepancy cause.
- Reset diagnostic SHA06d8ac97c46ab1108a5510df212b81ceea1e743f395fc5e1d8bcf0ea7cdceb09;
  waiting queueSHAf11b3118758a89df4bb7d4b1e819782fe1519c2f9520efec2eb125af66c563f2.

## Independent trained-repeat queued — approximately18:28 UTC

- Public HF dataset metadata still reports pinned revision cf5ef6748443ae72e9647f398487e98d421a7243
  and last modification2026-04-10. Read-only anonymous CLI queries found no
  updated environment revision; no data replacement or author contact.
- GPU fresh fixed20 repeat2 first episode return -58.32543484652, versus
  repeat1 -58.32662165193. Second episode still running. Quantify full logged
  trajectories after completion; do not equate a shared first step with parity.
- Auditing public_learner.py shows it resets immediately on training truncation,
  then again before evaluation at that boundary. Thus the single-reset fixed
  discrepancy must not automatically be blamed for the trained400 mismatch:
  its evaluation already follows two resets, and its first transition matched
  the original independent-final replay. Reset history and fresh-process
  numerical variation require separate checks.
- Added waiting tmux hg-pinball3d-fresh-trained-repeat-20260906, after the bounded
  double-zero diagnostic, for one complete200-action fresh R9 evaluation using
  the exact previously evaluated candidate.pt SHA0acb5c81adaeffbd437ebd681d01ffa9015c889ec9b06c991686b8f204b23a81.
  Default reset, same physical protocol, no optimization or score transform.
  No full3D training is queued. An initial launcher path/hash mismatch (numbered
  torch-save file versus candidate.pt serialization) was caught before launch;
  changed to exact candidate.pt and restarted only the waiting queue.

## Two fresh fixed-control processes complete — approximately18:31 UTC

- Both default GPU fixed20 processes exited0. First-episode raw returns
  -58.32662165193 / -58.32543484652; second-episode returns
  -58.34666196071 / -58.34664493788. Identical actions in all four episodes.
- Complete cross-process first-episode comparison: max reward error .00190991,
  max next-observation error .000097096. Second-episode comparison: max reward
  error .0000029810, max next-observation error .0000014305. These are bounded
  numerical comparisons, not statistical acceptance of a200-control learned
  trajectory, and do not explain the trained400 replay discrepancy by themselves.
- Audits pinball3d-gpu-fixed20-fresh-episode{1,2}-comparison-20260906.json bind
  full completed logs with hashes. Double-zero GPU diagnostic is now running;
  CPU8 second episode is at13/20 controls. Exact trained candidate fresh-repeat
  queue SHA718c77c7bbab920280cd8b6a56b5ded7326764e0e62febc015ea678fb8b7a016.

## Reset prefix / cavity physical logging — approximately18:40 UTC

- Double-zero diagnostic has34/40 recorded controls. Across its reset, first
  and second rewards are now exactly equal to the first episode, whereas the
  default single reset changed the first reward by.04916487. However, step2
  next observations still differ by.00011280179 across89 components, so this
  is not a complete reset/replay fix. All these early differences are confined
  to x=7 and8 downstream probes (45u,44v); cause not established. Full20
  comparison waits for the second episode to finish.
- Cavity optional raw probe recording and full-window velocity analysis were
  added to host baseline sources; no raw reward/policy input changes.15 tests
  passed. R9 is unchanged; CPU validation mounts a frozen post-R9 source.
- First CPU zero100 attempt exited1 before any CFD controls: unnecessary
  public_learner import required missing tensordict. Import moved into only
  train/evaluate modes; no new dependency installed. Failed output preserved
  with cavity-cpu-zero-probe100-import-exclusion-20260906.json. New v2 process
  runs CPU16–17 with16GiB cap, no GPUs/network, unchanged portable data bundle.
- Frozen v2 cavity_train.py SHA f17046bf74a0e82d0e12a283017ba4fedfbf81a7719056c3909614683edc13f0;
  cavity_reward.py bd67f375f6e088d00a5b906a48b92a76e8726cb4bd2a0d889c5b931befc28d4b;
  cavity_velocity_metrics.py 5dcbb5012c38346fe27c488bb5f93ba229b4e67d7f704d34d3681482ae343a0c.
- Task README/test scope now distinguish built/tested development images and
  native two-case integration from still-uncompleted full-suite performance.
  Public 3D cavity reconstruction ambiguities recorded separately; no generated
  mesh substituted for missing original assets.

## Double-zero completed / full trained replay active — approximately18:44 UTC

- Double-zero GPU diagnostic exited0 with returns -58.32664577300 and
  -58.29576646492. Complete within-process reset comparison still has maximum
  reward difference .00721604 and next-observation difference .000936151.
  Audit pinball3d-gpu-doublezero20-reset-comparison-20260906.json. It eliminates
  the first two reward differences, but is not a complete fix and has NOT been
  adopted in any official learner or evaluator.
- hg-pinball3d-gpu7-trained400-fresh-repeat-20260906 is running a full200-action
  replay of the exact original candidate with default reset. No training updates.
- Corrected cavity scaled400 pilot checkpoint c006f6f7a2084439f2ddef0c858525b6892272a12ebea7d7e4997a35318a051d
  exported in R9 without GPU. ActorSHA74957cf0ce279f7e6fa90a7a97f739bd455c69f453fdf496a3f493710881113d,
  all64 synthetic observation actions exact. This is packaging evidence only.
  Waiting CPU16–17 queue hg-cavity-cpu-trained-velocity-20260906 runs its portable
  policy after the current zero100 check, with the same reconstructed calibrated
  data and optional raw probe logging. It cannot overlap that CPU allocation.
  The400-step pilot is not a full-budget performance result or shared submission.

## User-requested status snapshot — 2026-09-06 18:46 UTC

- Corrected training completed episode counts: Pinball DDPG70,200/100,000;
  TD3scaled62,400/100,000; PPOauthornorm75,400/200,000. DDPG latest68k eval
  return-346.01530, best56k -130.15576; TD3 latest60k -301.35976,best56k
  -218.49245. These are not final independent-seed results or paper matches.
- Cavity publicPPO48,700/100,000; scaledPPO15,000/100,000. Latest evaluations
  -79749.21167 and-70400.15840 respectively, still worse than zero control.
  CylinderPPO8,200/500,000; first10k development evaluation still pending.
  JAXCPU unscaled4,640/5,000; GPU Eq34alternative1,600/5,000.
- CPU cavity zero100 velocity recording completed: rawreturn-6802.668616663203
  versus prior GPU -6802.668616663734 (difference5.31e-10). This checks unchanged
  physical reward across the CPU/backend and optional logging path, not paper
  performance. Full34-probe velocity metrics produced. The exported scaled400
  resident policy is now undergoing its CPU100-action closed-loop replay.
- CPU3D fixed40 controls completed, episode returns-58.32518975930 and
  -58.34663233681; process still in normal native teardown at this snapshot.
  Full200-action trained3D fresh replay is initializing/running GPU7.
  No full3D training or whole-suite performance acceptance.

## User-approved staged baselines — 2026-09-06 22:27 UTC

- First reproduce the five case-matched paper-curve cases, then adapt a
  verified method to the other selected cases as organizer baselines. This
  supersedes the requirement for a recovered published policy score on every
  resource variant. Full47 membership/shared-model contract remain unchanged.
  See STAGED_BASELINES.md. No agent or full-suite adaptation was launched.
- Existing corrected Pinball2D and CylinderRe3900 runs continue unchanged.
  DDPGseed42 and TD3scaledseed42 have completed100k controls and fresh replays;
  their seed43 successors are active. These are not accepted paper matches.
  The 3D fresh-trained repeat returned-675.42396676145, near the earlier fresh
  replay-675.47656 but not the original training-process-649.48091. This does
  not resolve the reset-history discrepancy; no long3D training was launched.
- Added CylinderRe1000 public-default65-probe support, explicit paper350/200
  coupling, standard-release mount option and checkpoint/replay release identity
  guards. Original running/queued source snapshots and data were not changed.
- GPU7 zero-control run hg-stage1-cylinder1000-zero-s42-20260906 completed
  200 controls/exited0: meanCD1.53077750363 vs SI1.533 (-.14498%), sampled
  CL half peak-to-peak1.44832423820 vs1.449 (-.04664%). RMSCL1.04500318889 is
  a different statistic. Episode wall125.34868sec. No paper-policy claim.
- New hg-stage1-cylinder1000-ppo-check-s42-20260906 is running on GPU7:
  from-scratch public-sourcePPO,400 training controls, eval_every400,
  bootstrap time limits and fresh final/best replays. It mounts the verified
  hydrogym-public-500gb-v1 data read-only. Actual protocol records130
  observations,1 action,omega1,350 substeps,horizon200,max_control.03.
  Baseline imageID sha256:1eb3775b0b4997b2b9fee6a68a8810f687879978a1fab7f7976d53f62266842d.
  Full400k training is not queued automatically; review the short run first.
- Validation:12 baseline unit tests passed; all9 public/default/tuning recipe
  variants passed real optimizer updates and checkpoint replay on synthetic
  transitions, including rejection of changed release identity; all65 task
  tests passed in offline R9 with current source read-only. No image rebuild,
  upload, Git push or new external data. HF CLI reconfirmed the pinned revision.

## Five-case execution and measured comparison — 2026-09-07 02:04 UTC

- User requested execution and a paper/public-result comparison on the five
  alignable cases. See STAGE1_RESULTS_20260907.md and the hash-bound snapshot
  reproduction/stage1-comparison-20260907-initial.json (02:01 UTC). Every selected
  score comes from a complete frozen-policy fresh-process evaluation; three are
  400-control pilots, not converged full experiments. No paper match accepted.
- CylinderRe1000 pilot and SquareRe200 pilot matched all reward/action/probe/force
  values exactly over 200 controls after checkpoint reload. The existing
  Re3900 immutable20k checkpoint also matched its training-process trajectory
  exactly in a fresh GPU6 process. Measured last-half CD respectively1.507044,
  1.500326 and1.127054. Pinball2D DDPGseed42 best80k gives .250396;
  Pinball3D pilot400 second fresh replay gives2.921955.
- Launched standard-release from-scratch full PPO jobs:
  hg-stage1-cylinder1000-ppo-full-s42-20260907 onGPU7 (400k controls),
  hg-stage1-square200-ppo-full-s42-20260907 onGPU0 (600k controls).
  At02:04 they have logged1000 and400 training controls, respectively. Both
  are live; existing Re3900/Pinball queues and unrelated processes unchanged.
- Square native support now uses40 public-default probes in y-major order,
  SI225substeps/200controls,omega.2, pinnedtwoindependentjets/.03. Force logs
  convert raw lattice values to coefficients consistently with native reward.
  Its no-control CD1.489440 vs SI1.469 (+1.391%); RMSCL.426960 vs.391 (+9.197%).
  Original meshes/restarts/configs not modified; old paired-jet author setup
  remains a documented difference, not silently adopted.
- GPU1 runs hg-stage1-pinball3d-trained-reinit2-s42-20260907-r2: two complete
  episodes with the same400-control checkpoint to investigate reset history.
  No full3D training or new reset convention. GPU6 runs a bounded1200-control
  ddpg-slow/.5 Pinball2D reconstruction check; no automatic full-run successor.
- Two attempted diagnostics were rejected before rollout (unsupported timing
  flag in frozen older source; new release guard on a legacy checkpoint).
  Retained their logs and used distinct corrected-r2 runs, without weakening
  current identity checks or retrospectively attaching release labels.
- Tests:15 baseline adapter tests,5 comparison-report tests,65 task tests
  passed; shell/compilation checks passed. All8GPUs show task activity at02:04;
  /mnt/workspace has1.5TiB free. No Agent, push, upload or author contact.
