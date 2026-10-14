# WN2 model delivery contract — implementation under validation

Deliver one portable model directory at `/workspace/submission`, including
`infer.sh`, trained parameters, configuration, preprocessing and all inference
resources. The limit is 8 GiB and 10,000 regular files; symlinks are forbidden.
Training checkpoints may also contain optimizer state, but inference needs no
training data, pretrained base mount, original workspace or undocumented cache.
The declared model runtime is the pinned WeatherNext/JAX image; inference has
one allocated GPU per case; no specific GPU model is required.

The package may be read at any moment. Replace it as a whole: assemble it elsewhere on the same
filesystem and swap it in with one rename, as the starter's `train.py` does after packaging.

## Invocation

```bash
bash /relocated/submission/infer.sh INPUT_DIR OUTPUT_DIR
```

`INPUT_DIR/request.json` contains:

```json
{
  "schema": "wn2-inference/v3",
  "cases": [{"id": "new_case", "file": "context.nc"}],
  "members": 56,
  "lead_hours": [12, 24, 36, 48],
  "seed": 0
}
```

IDs are distinct strings of 1–100 ASCII letters, digits, `_` or `-`. File paths
are relative and stay inside the input directory. The implementation must
support unseen dates, renamed IDs, arbitrary case order and positive 12-hour
lead multiples through 360 hours, in increasing order. V3 requests one case
and consecutive new leads per invocation; v2 can batch cases and request lead
subsets. Support one through 56 ensemble members. The benchmark uses 56 members and 30 leads.

Each NetCDF context contains one `batch`, two `time` coordinates `[-6h, 0h]`,
`datetime(batch,time)`, the exact 721×1440 global grid and the 13 native pressure
levels. The initialization is the final datetime. Latitude increases from
−90 to 90; longitude increases from 0 to 359.75. Dynamic native input fields
use `batch,time,[level],lat,lon`; static geopotential/land mask omit time/batch.
Calendar forcings may be derived from timestamps. No future weather or labels
are provided. The task's input materializer documents isolated-cell repairs
and the SST land mask. Forecasts must use the delivered model autonomously.

## Lossless continuation (v3)

The default disk-bounded verifier requests one case and a consecutive subset of
12-hour endpoints, then pauses the model for scoring. An initial request starts
at hour 12. A later request adds, for example:

```json
"resume": {"directory": "resume", "lead_hours": 24}
```

Its first requested lead must be 36. `INPUT_DIR/resume/member-NNN/` contains the
same member's state produced by the preceding invocation. Continue from it;
do not regenerate prior forecast steps. For every requested member, write
`OUTPUT_DIR/state/member-NNN/` when the final requested endpoint is before hour
360. State is opaque to the verifier, must contain only regular files/directories,
and must stay within the planner's per-generation size bound (about 49 GB for
the canonical 56-member grid). The stock implementation stores two context frames
losslessly, retaining forcing values and all native input variables; its state
receipt binds package, initialization, member, seed and completed lead.

Only the requested lead window is emitted as forecast arrays. Those fields must
be float32, exactly the declared weather and precipitation fields; do not append
past forecasts to the output. Member indices and random streams must be independent
of window size, request ID and invocation order. The stock seed is keyed by
initialization, member and six-hour step, so restarting does not reset the stream.
The 12-hour rain total still sums the two underlying six-hour intervals.

A continuation receives the original context and its own preceding state. It
never receives truth, scores or cyclone projections. The process namespace exits
before the verifier reads outputs or scores a window. The old state generation
is removed only after the next window has completed successfully.

New stock packages implement v2 and v3. Existing stock weights can use
`environment/weathernext/wn2/upgrade_inference.py` (task repository) without retraining. A custom submission must implement
this continuation interface itself. Legacy v2 full-case execution is available
only through explicit `--legacy-case-mode`, with its larger disk requirements.

## Outputs

Write `OUTPUT_DIR/ID-member-NNN.zarr` for each member of each input ID,
using consolidated Zarr v2; `NNN` is the zero-padded member index.
Weather and precipitation fields are listed in `protocol.json` and
`precipitation-protocol.json`; all are required. Arrays have dimensions
`case,lead,member,[level],lat,lon`, with one case and one member per store. Coordinate values
and ordering must match the request, native grid and levels exactly.
`valid_time(case,lead)` must equal initialization plus lead. Members are 0-based.
Units must match the canonical protocol units. Native cyclone/SST/100m outputs
may be retained separately; the model's training schema is unchanged.

Every scored prediction must be finite, including cells with missing truth.
Unwritten chunks retain NaN and fail. Missing cases, wrong IDs, reordered
coordinates, malformed metadata and missing fields fail explicitly. Only
primitive numeric array types and the documented safe compression codecs are
accepted. Each member store has a 160-GiB decompressed bound and a 256-MiB chunk bound.

## Independent scoring and reward

`verify_case.py` copies the model to a fresh directory and runs it in an offline
PID/user/mount namespace with only the declared runtime, model, context and
output mounts. Truth and training data do not enter this process. Scoring starts
after the model's process namespace has exited. The verifier, rather than the
submission, computes scores from the predictions.

Report empirical WeatherBench-X CRPS (`fair=False`), ensemble-mean RMSE, spread
and pooled diagnostics by variable, level and lead. The weather reward is
`1/(1+normalized_CRPS)`, where each CRPS is divided by its frozen training-only
HRES standard deviation. Average equally across pressure levels and leads
within each variable, then equally across the ten weather variables. Use only
native-cell scores in the primary reward; larger is better. This is a
dimensionless quality measure, not a mixture of raw physical units.

Precipitation is a separate ERA5-verified panel; it is not folded into the
weather reward. Each output is the 12-hour accumulation ending at its lead,
formed by summing two adjacent six-hour model intervals. The evaluator also
reports 24-hour accumulations, SEEPS, extremes, spherical pooling, power spectra
and cyclone tracks (position verification through nine days). Keep all absolute scores and paired
comparisons against the trained starter, persistence/dry precipitation and
the available reference evaluations. A failed prediction receives zero reward with a
failure record; setup/scorer errors must remain identifiable and cannot be
presented as measured model skill.

## Training and visibility

Initialize the full native WN2 parameters randomly and train all parameters;
no pretrained initialization or LoRA. Preserve the shortened staged curriculum,
native two-sample training CRPS and full target schema. One model seed and a
small seasonally selected training slice are deliberate reductions. Training
statistics and cyclone calibration use only declared training years.

The provisional split is training through 2020, development 2021, test 2022.
The 32 training windows are selected before training. Four development dates
are visible; the eight-date test panel is fixed before candidate training and is
not used for model selection. This is a retrospective public-data task; the
operator's preparation has access to public truth, so it is not a claim that
the source data are secret. Candidate inference is isolated from truth.

The starter ships the reference schedule selected from operator-side timing
measurements. Agents start a complete training run directly; profiling and
adaptation diagnostics are not prerequisites. Independent evaluation remains
separate from training. Final reference acceptance is still pending completion
of the trained package's full independent evaluation and verifier checks.
