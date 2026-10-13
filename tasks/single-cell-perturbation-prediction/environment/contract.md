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

A missing file, wrong gene order, wrong cell count, a request without predictions, negative or
non-finite values, or a timeout fails the whole context (score 0 for that context); the other
contexts are still scored.

## Scoring

Per context, the pinned `cell-eval` 0.8.2 Generalist panel is computed on (predictions vs.
private responses, both with the same basal cells as control): DES (`overlap_at_N`),
PDS (`discrimination_score_l1`), MAE, Pearson-delta, Spearman of significant-gene LFC, AUPRC of
DE recovery, and Spearman of effect size (`de_spearman_sig`). Each metric is normalised
against the no-effect baseline (basal cells resampled per request) with cell-eval's rule:
`(m - m_base)/(1 - m_base)` for higher-is-better metrics, `1 - m/m_base` for MAE, clipped at 0.
The context score is the mean of the seven normalised metrics.

Each test context's 300 requests are split into a **leaderboard** subset (100 requests; reported
separately) and a **final** subset (200 requests; the reward). Subsets are scored
as self-contained cell-eval runs. **Reward = mean over the test contexts of the final-subset
score.** The report also contains per-stratum breakdowns (effect-pattern strata, seen vs.
withheld targets). The development scorer `evaluation/cellscore.py` is the same code path.

## Rules on the package

- One generic model: `basal + targets -> predictions` for any context, panel and target list.
  Lookup tables keyed on the test context, on request IDs or on the shipped test target lists,
  cached predictions, and any use of held-out responses are forbidden.
- Falling back to "no effect" for targets or contexts the model cannot handle is legal; it scores
  as the baseline for those requests.
- The package must be reproducible from `/workspace/research` (config, seed, data view, training
  script) as documented in `instruction.md`.
