# Model delivery and inference contract

The graded artifact is the directory `/workspace/submission`: a self-contained model package
(inference code, configuration, all learned weights, preprocessing) and a concise `README.md`.
The verifier copies it into a fresh container built from the task's verifier image, with no
network and no access to the development workspace, and calls it once per test context.

The package may be read at any moment. Replace it as a whole: assemble it elsewhere on the same
filesystem and swap it in with one rename, as `/workspace/baseline/publish.py` does.

## Invocation

```
bash infer.sh INPUT_ROOT OUTPUT_ROOT
```

- Runs as uid 10001 (`agent`), working directory = the package root, `CUDA_VISIBLE_DEVICES`
  set to exactly one GPU, `PIP_NO_INDEX=1`, `HF_HUB_OFFLINE=1`, no network.
- Runtime: the verifier image (`tests/Dockerfile`) is the development image's runtime; nothing
  else is installed. Vendor any extra pure-Python code inside the package.
- The frozen SE-600M release (`se600m_epoch16.ckpt`, `config.yaml`, `protein_embeddings.pt`,
  licences) is available read-only at `/opt/se600m` in **both** the development and the verifier
  container (`$SE600M_DIR`; in development it is the same directory as `/workspace/data/se600m`). A package that uses SE at inference should read it from there
  rather than copying 11.5 GB into the submission; any other weights must ship in the package.
- Budget: 60 minutes wall-clock per context, 16 GiB output limit. The package must be
  relocatable: no absolute paths into `/workspace/data` or the development workspace, no
  undocumented caches.

## Inputs (`INPUT_ROOT/`)

| File | Content |
|---|---|
| `basal.h5ad` | Non-targeting (unperturbed) cells of the context. `X`: raw UMI counts, CSR float32. `obs`: `batch` (opaque labels), `total_counts`, `target_gene == "non-targeting"`. `var`: index = gene symbol, column `ensembl_id`. The gene panel is the context's own and differs between contexts. |
| `requests.jsonl` | One JSON object per line: `{"request_id", "context", "perturbation": {"type": "crispri", "target": <symbol>}, "n_cells", "seed"}`. Target symbols are HGNC-style symbols; a target may be absent from `basal.var` (not expressed / not in the panel). |
| `context.json` | `{"context_id", "chemistry", "n_genes", "n_basal_cells", "n_requests", "perturbation_type"}`. Nothing identifies the cell line. |

The verifier may use different contexts, gene panels, target lists, request counts, `n_cells`
values and request orders than the ones shipped for development. Never hard-code them.

## Output (`OUTPUT_ROOT/predictions.h5ad`)

- Exactly `n_cells` rows per request, `obs["request_id"]` set on every row; any row order.
- `var` in exactly the input `basal.var` order (same symbols, same order).
- `X`: predicted post-perturbation expression as **non-negative count-scale values** (raw UMI
  counts or a continuous count-like matrix on the same scale as `basal.h5ad`). CSR float32
  preferred. The scorer applies `log1p(counts per 10k)` to predictions and ground truth alike.
  Do not output log-normalised values.
- Optional extra `obs` columns (e.g. `target_gene`) are ignored.
- The HDF5 file must contain its own data: external links, symbolic HDF5 links,
  virtual datasets, external raw storage, and a filesystem symlink as the output file
  are rejected before loading. A sealed output must not resolve differently in the
  inference container and the private scoring container.

A missing file, wrong gene order, wrong cell count, a request without predictions, negative or
non-finite values, or a timeout fails the whole context (score 0 for that context); the other
contexts are still scored.

## Scoring

The primary `score` (also written identically as `reward` in `reward.json`) is the equal mean across
the three held-out contexts (`near`, `mid`, `far`) of the V3 dual-skill context score, combining
calibrated response-amplitude accuracy (`amplitude_skill`) and perturbation-specific identification
(`identification_skill`).

Within each context, the grader normalises each cell to counts per 10,000, takes `log1p`, and
averages by perturbation target over the final roster of targets with resolvable Ensembl identities.
The verifier splits measured cells per target into two halves across five fixed splits and both
directions (`5 × 2 = 10` evaluations):

1. **Signed amplitude skill (`amplitude_skill`)**: One half supplies gene weights from squared,
   min–max scaled absolute one-versus-rest $t$-statistics (excluding the directly perturbed target
   gene when present in the output panel); the other half supplies the target mean to predict.
   Let $L$ be the candidate WMSE, $N$ the WMSE of the generic equal-target mean response across
   perturbations, and $P$ the same-target experimental split-half repeat WMSE:
   `amplitude_skill = (N − L) / (N − P)`.
2. **Signed identification skill (`identification_skill`)**: On the non-target gene panel (all
   roster target genes excluded), let `NIR` be the Euclidean nearest-interventional retrieval rate
   (with exact ties receiving `0.5` credit) against the reference half-means, and `repeat_NIR` be
   the split-half repeatability NIR between complementary measured halves (`repeat_NIR > 0.5`):
   `identification_skill = (NIR − 0.5) / (repeat_NIR − 0.5)`.
3. **Order of operations (average signed skills BEFORE zero-clipping)**:
   `signed_context = 0.5 * amplitude_skill + 0.5 * identification_skill`, and then
   `context_score = max(0.0, signed_context)`. (`amplitude_skill` and `identification_skill` are
   never clipped individually before averaging.)
4. **Overall task score**:
   `score = reward = clip(mean(context_score for c in ("near", "mid", "far")), 0.0, 1.0)`.

The v2 WMSE-only reward (`clip(amplitude_skill, 0, 1)`) and the seven cell-eval 0.8.2 Generalist
metrics remain available as diagnostics in `report.json`. Each context has 300 requests (100 in the
leaderboard subset and 200 in the final subset); inference must predict all supplied requests. The
`run_loco.py` development entry point applies the exact same V3 context formula (`score_context_v3`)
to each held-out training context's public `dev_split/`.

## Per-request query independence (two-pass verification gate)

Each perturbation request in `requests.jsonl` must be predicted **independently of batch
composition, request ordering, or what other target genes appear in `requests.jsonl`**:

- Seed any stochastic basal-cell resampling or generative sampling **deterministically per
  `request_id`** (or using the request's `seed` field, as all supplied baselines do) without
  consuming a shared sequential RNG across the request list and without cross-query shrinkage,
  batch-mean centering, or transductive adaptation across `requests.jsonl`.
- During isolated verification, the label-free `main` runner executes **two independent passes** per
  context in separate fresh processes (with isolated ephemeral `HOME`/`TMPDIR`):
  1. **Pass 1**: full `requests.jsonl` manifest (`predictions.h5ad`);
  2. **Pass 2**: a deterministic shuffled proper subset of `requests.jsonl` (`predictions_subset.h5ad`).
- Trusted `grader.py` compares the per-request mean `log1p(CP10k)` expression profiles between Pass 1
  and Pass 2 via `query_independence.compare_request_predictions(..., atol=1e-4, rtol=1e-4)`. Any
  submission that fails Pass 2 or changes any request's mean `log1p(CP10k)` profile beyond
  `atol = 1e-4, rtol = 1e-4` is rejected by the hard gate with `score = 0.0` and `reward = 0.0`.
- `evaluation/check_submission.py` runs the same two-pass subset check locally so you can verify
  compliance before publishing.

## Rules on the package

- One generic model: `basal + targets -> predictions` for any context, panel and target list.
  Lookup tables keyed on the test context, on request IDs or on the shipped test target lists,
  cached predictions, and any use of held-out responses are forbidden.
- A model may return a basal-cell forecast for an unsupported target. The verifier scores that forecast
  against the same two calibration anchors as every other prediction.
- The package must be reproducible from `/workspace/research` (config, seed, data view, training
  script) as documented in `instruction.md`.
