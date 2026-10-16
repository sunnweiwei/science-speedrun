# Fixed primary evaluation metrics

**`hydrogym-physical-metrics/v1`, adopted 2026-09-10.** This covers exactly the
39 cases of `hydrogym-public-research-v2`. [metrics.py](metrics.py) contains the
complete, explicit case-to-metric assignment. `profiles.json` exports the
per-case definitions and `metrics_sha256`; tests require exact release coverage.
Changing a metric, unit, target, normalization or window requires a new version.

These are our reporting definitions, grounded in the native control objectives,
not a claim that the paper defines one universal scalar. Native training rewards,
physics, source data, case membership and policy/v1 delivery are unchanged.

## Primary metrics

| Cases | Count | Metric / unit | Direction | Window |
| --- | ---: | --- | --- | --- |
| All Cylinder, SquareCylinder, Pinball, Cube and Sphere cases | 19 | Mean total drag coefficient CD; dimensionless | Lower ↓ | Last half |
| All NACA0012 cases without Gust | 7 | Mean instantaneous CL/CD; dimensionless | Higher ↑ | Last half |
| All NACA0012Gust cases | 9 | Mean weighted load error abs(CL−CL₀) + ω·abs(CD−CD₀); coefficient units | Lower ↓ | Full episode |
| Nozzle2D_coarse, Nozzle2D_fine | 2 | Mean absolute thrust-angle error; degrees | Lower ↓ | Full episode |
| Kolmogorov_2D | 1 | Mean fluctuation kinetic energy TKE; native nondimensional units | Lower ↓ | Last half |
| Channel_3D_Retau180 | 1 | Mean native wall-shear-stress observable; native nondimensional units | Lower ↓ | Last half |

Cavity and other historical experiments are outside these 39; do not silently
insert their metrics into this release's table.

## Exact computation

For a complete H-control episode, last half means controls `floor(H/2)+1 ... H`
inclusive; full episode means `1 ... H`. The reset observation is excluded.
MAIA native H=201 uses 101–201; earlier paper-derived H=200 runs use 101–200
and remain separately labeled experiments. No adaptive window trimming or
terminal-value/training-curve substitution. The full gust/nozzle window retains
the early disturbance/target response, not just its recovered final state.

Control samples have equal weight. CD, L/D, loads, wall shear and angle use the
native post-control observable. Kolmogorov `mean_tke` already averages internal
CFD samples within each control. Average complete episode metrics equally over
the frozen initializations. Current native replay uses seed 123 and source flow
fields; repeated deterministic resets are not independent flow samples.

- **CD:** sum over all bodies, not their mean. For raw MAIA forces, use source
  density=1, reference velocity=Ma/sqrt(3), and reference area D·span; Sphere
  uses pi·D²/4. Pinball `info.forces` already contains coefficients: never
  normalize twice. Preserve signed CD as in our earlier comparison table.
- **L/D:** mean of instantaneous CL/CD, not mean(CL)/mean(CD). Preserve the
  source's operating domain CD >= 1e-6 and CL >= 0 at every scored sample.
  Otherwise return `invalid_operating_regime`, value null, not the artificial
  -100 training penalty as a physical L/D value.
- **Gust:** CL₀, CD₀ and ω are the pinned source YAML's unperturbed targets and
  weight, exported per case. All nine current ω values are 0.25. These targets
  are not the candidate's forces or the uncontrolled-with-gust test trajectory.
- **Nozzle:** `mean(abs(target_angle−thrust_angle))*180/pi`, using unnormalized
  native radian angles and the source's absolute difference (no changed wrapping
  convention). Report degrees, not cumulative negative radians.
- **TKE:** source spatial mean of `0.5*((u−mean(u))²+(v−mean(v))²)` averaged
  over each control interval. Exclude the training reward's L1 action penalty.
- **Channel:** source `wss_compute(4,U,z)`, the plane mean of
  `1.9e-3*(U[:,:,4]−U[:,:,0])/(z[4]−z[0])`. This is the native shear observable,
  not a newly named friction coefficient Cf. Its pinned reward is exactly
  `−wss` without another penalty; extracting via that verified identity leaves
  the hash-frozen actuator-clock adapter unchanged.

## Reporting contract

Default columns:

`Task / experiment | status / training progress | metric / unit / direction |
uncontrolled | Final/latest checkpoint | Best checkpoint | paper reference |
configuration/evaluation notes`

Declare the window in the row or legend. Every compared number within a row
must use the same metric/unit/statistic. Paper configuration differences are
allowed and must be stated; do not suppress a reference merely because the
mesh, sensor layout, seed, budget or algorithm differs. Another physical metric
or averaging window is not a same-metric reference. Label approximate paper
curve readings and cite the source figure; do not call them exact seed means.

Uncontrolled uses zero physical control under the same local evaluation profile;
it is not a random policy or the gust's unperturbed target. If sharing one
uncontrolled run across training seeds, say so. Improvement percentages are
optional explanatory derivatives, never replacements for the primary value.

Read `results.json -> cases -> CASE -> primary_metric` for `id`, `version`,
`name`, `units`, `direction`, `window`, `status`, and `value`. Episode values
are in `episodes[].primary_metric`. The trusted evaluator computes these,
not candidate code. `return`/`mean_return` remain diagnostic fields only.

The Harbor verifier and published specialist table report physical metrics.
Missing physical data stays missing: do not substitute training R,
terminal info or another experiment. Historical R-only artifacts need replay.
Previously measured paper-CD results remain valid for their named experiments,
not relabeled as Research v2 native results.

Best keeps its checkpoint identity and recorded selection rule. Historical
Best was selected by development R, not by post-hoc lowest CD; say so. Future
research should track these fixed objectives while preserving action bounds
and retaining stability/control-cost diagnostics outside the main score table.

## Implementation and acceptance scope

Per-case definitions, extraction, windows, units, coverage and report plumbing
have CPU tests. The published specialist snapshot has complete paired independent
physical results for all 39 cases. See
`/workspace/references/results/results.json` (repository: `environment/references/specialists/verified/results.json`) for immutable evidence hashes and scope.
These specialists are not a jointly trained model or multiple independent seeds.
Running experiment snapshots are untouched.

This version does not invent a mixed-unit aggregate. The legacy normalized-R
grader in `evaluate.py` is not the current primary metric; its historical
aggregate must not be presented as the physical benchmark score. Cross-case
normalization requires a separately versioned decision and measured anchors.

The active Harbor entrypoint is `harbor_verify.py`: it runs the native evaluator
and emits `CASE/metric_id: physical_value` in reward.json. Follow the per-case
directions above. It exports no fake aggregate and no inferred paper anchors.
A case whose policy fails (protocol error, a simulation that turns nonfinite after the policy
has acted, or an invalid operating regime) has no physical value: reward.json flags it as
`CASE/failed: 1` and counts it in `failed_cases`, and every other case keeps its value. Only when
no case produces a value does reward.json hold `invalid_submission: 1`. Invalid state before the
first action, a broken solver and missing cases are infrastructure faults: labeled, nonzero exit.

Source basis: HydroGym commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`,
`hydrogym/maia/envs/{cylinder,square_cylinder,cube,sphere,pinball,naca0012}.py`,
`hydrogym/jax/envs/{channel,kolmogorov}.py`, `hydrogym/jax/utils/utils.py`, and
`hydrogym/jaxfluids/envs/nozzle.py`. Aggregation windows are organizer choices,
not recovered unpublished paper settings.
