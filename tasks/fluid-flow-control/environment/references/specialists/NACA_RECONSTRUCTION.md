# NACA0012 gust audit (not a reproduced control baseline)

The paper Fig.2 includes 3D NACA0012, Re1000, incidence 20 degrees, with a
transverse gust. Pinned `NACA0012Gust_3D_Re1000_AOA20` assets are present and
hash-verified (about 34 GB, mainly the restart). The grid header has 72,790,355
cells, matching the paper table count; that alone does not verify all physics.

The supplement Eq.37 matches the current native reward: negative absolute lift
deviation minus omega times absolute drag deviation. The specified omega=.25
agrees with YAML, whose unperturbed reference CD=.402 and CL=.833 still require
independent physical checking. Source properties specify Ma=.2, Re1000, span4,
a unified two-solver configuration (solver0 is cumulant LB, solver1 is the
MAIA_POST_DATA probe/postprocessing component) and three leading-edge jets.
This resource is not a coupled LB/finite-volume flow solver; the generic
MAIA_UNIFIED label must not be interpreted that way.

The current YAML uses 160 CFD substeps/action and max_episode_steps=200 (201
native controls). The supplement gives a case-dependent range 225–1000 but no
exact figure interval. The older author class provides 640 steps/action and a
25-action default for (Re1000,3D). Its 4x3x3 default wake probes span x=1..8,
y,z=-.75...75 with u/v measurements. These are a public-example reconstruction,
not recovered final figure sensor metadata. The short window is not asserted to
cover the full gust interaction until force and probe traces are examined.

`native.py` adds only zero/smoke diagnostics for this case. It preserves native
Eq.37 and source actuator bounds; no NACA learned-baseline claim is enabled.
`author-paper` uses the explicitly documented 640/25/.25 settings. Native force
info is raw lattice force here (unlike Pinball), so only diagnostics are converted
to coefficients; scalar reward remains unchanged. The two-solver launch and
memory/output requirements still need a real solver run before training.

## First native launch and restart-stage audit — 2026-09-06 12:10 UTC

The73M-cell solver has started on one H100, using about62,599MiB GPU memory.
Grid/window construction alone took about800seconds on one solver rank.
The first force sample is CD=.39052, CL=.68388. The two-control smoke check is
still running; no complete gust rollout or learned baseline is claimed.

The folder contains both `properties_run_gust.toml` and `properties_run.toml`.
Only the former has gust injection parameters (factor2, duration16000), starts
from200000 and advances18500 steps; it omits the control jet setup. The latter
starts from218500 with jets and no new injection. The actual downloaded restart
header is218500/Re1000/Ma.2, generated in a directory ending in
`3D_Re1000_AOA20_gust`, with four original solver domains. This is consistent
with a preparatory gust-generation stage followed by control of the already
perturbed field. That interpretation is an inference, not exact paper-launch
provenance; the complete physical force/transport window still needs checking.
It would be incorrect to switch blindly to the gust-generation properties and
reuse a restart from the wrong stage. The source header/configuration audit is
`naca-source-restart-audit-20260906.json` in the reproduction directory.

## Completed short window and extension — 2026-09-06 13:00 UTC

The two-control smoke and25-control zero episode both completed. The latter
returned-8.1426969819, meanCL1.0932136, RMSCL1.1148515, meanCD0.5786013,
wall2597.9seconds including initialization. No learned policy was used.
Its force trace has not reproduced the full Fig.2c gust peak.

The inlet is8chords upstream. The gust-preparation run spans18,500CFD steps,
while this short control window adds16,000. That timing makes incomplete
gust transport a plausible concern, not a proved diagnosis. A new zero-only
diagnostic explicitly overrides the horizon to50 controls (32,000CFD steps)
with the same640-step interval and native restart. Compare its first25 samples
with the completed short run and inspect the later lift/probe history before
choosing a training/scoring window. It is not silently a new paper protocol.
Output: `hg-naca3d-gust-zero50-20260906`, using GPU2 alone.

## Completed extended window — 2026-09-06

The 50-control zero test completed with raw return -19.999859165985217,
mean CD .5926561004, mean CL 1.1737847027, RMS CL 1.1969213591,
and wall time 4465.99 seconds including initialization. Peak CL was
1.5447344062, not the approximately 4.4 peak visually read from Fig.2c.
Simply doubling the public example window therefore did not recover that
paper trajectory. No trained NACA result is claimed.

The first 25 steps closely reproduce, but are not bit-identical to, the earlier
25-control run: maximum absolute differences were CD 7.81147e-7, CL 1.21465e-6,
reward 1.25392e-6, and observation 1.94311e-5. The first difference appeared
in the third control's next observation. Preserve these measured differences;
do not treat a failed exact-equality assertion as exact replay evidence.

There is also an unresolved reporting-convention discrepancy: Fig.2c's reward
curve is positive (roughly 200--250 late in training), whereas the article's
explicit gust reward, supplement Eq.37 and both inspected source classes use
negative absolute force deviations, hence every raw reward and its unshifted
sum are nonpositive. No public plotting transform was recovered. Preserve raw
physical reward and lift/drag trajectories; do not add an arbitrary offset or
flip the reward sign to reproduce the displayed axis.
