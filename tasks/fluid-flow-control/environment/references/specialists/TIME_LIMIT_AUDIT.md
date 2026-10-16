# Time-limit reconstruction audit, 2026-09-06

The released author's native environment ends when `iter > max_episode_steps`.
Its TorchRL wrapper passes native `done` through and applies
`StepCounter(max_episode_steps)` externally. For a 200-control limit this
produces `done=True, terminated=False, truncated=True` at control 200. The
value target includes the critic's next-state value; GAE stops crossing the
episode boundary. The same pattern applies to the 100-control cavity example.

Our earlier manual adapter instead set the native limit to 199 (99 for cavity)
to obtain the correct number of CFD controls. This made the last transition
terminated and suppressed value bootstrapping. The rewards and control count
were unchanged, but the learner semantics differed from the public example.
The source comparison and executable TorchRL tests confirm this difference;
they do not establish that it accounts for the observed performance gap.

Use `--bootstrap-time-limits` for new public-example reconstructions. The
adapter leaves the native limit at the intended horizon, applies external
truncation, and rejects unexpected native completion instead of masking it.
The flag is recorded in both protocol and checkpoint configuration and carried
into independent evaluation. Missing flags in historical artifacts mean the
legacy behavior, which remains available for exact historical replays.

Existing runs and archived scores are not rewritten or pooled with corrected
runs. All remain non-certified reconstructions until physical performance is
independently validated against the paper. The audit is based on the pinned
author revision `20a483ecc6ea711f6b380fff05f212ca4d7e2c67`,
`torchRL_examples/maiaGym_TorchRL_wrapper.py` and
`maiaGym/maia_env_core.py`, and pinned current native `env_core.py`.

`test_time_limits.py` compares all 200 transitions with the actual TorchRL
StepCounter and verifies the GAE target. `test_public_learner.py` also rejects
checkpoints evaluated with different time-limit semantics. Synthetic tests
are not paper-performance evidence.

## Shared starter and external evaluator

The shared MAIA path now uses case-level `episode_end="time_limit"` with native
limit equal to the reviewed horizon. `run_maia.make_env` applies the same wrapper
for shared training and independent replay. Omitted fields retain historical
`native` semantics; no existing release or running experiment is hot-modified.
Unexpected native completion, an unknown convention, or a step after timeout
without reset is rejected. Shared resume configuration includes the selected
boundary conventions so a legacy model cannot silently continue with different
critic targets.

`hg-shared-bootstrap-cpu-check-20260906` passed all29 task tests using mounted
host implementations in the R7 runtime, with no skips. Its actual SB3 rollout
checks constant reward-1, value5, gamma.99: first199 rewards remain-1 and the
time-limit transition is stored as3.95 after SB3's bootstrap correction. This
is stronger evidence than flag inspection alone, but is still synthetic.
The R8 image tests exercise the copied runtime independently. A queued native
two-case train/resume/relocated-replay test uses R8 runtime implementations and
a separately frozen diagnostic launcher; it is not a full benchmark score.
