# Materializing the pde-surrogate data (controller and maintainers)

Prepare data on an operator host with network access, then hand the prepared directories to
Harbor: `TASK_DATA_DIR` (agent side, each entry mounted read-only under `/workspace/data`) and
`TASK_PRIVATE_DIR` (verifier side). Harbor does not run any of this automatically. This file is
not copied into the agent's workspace.

## Materialization

Verifier-only preparation (just the 54 files the held-out cases reference,
63 GB: the corpus files with held-out tails or test sets, 30 Well test files,
8 PDEArena shards):

```bash
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" \
  --manifest pde-surrogate/environment/data/split.json --sources -
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" \
  --sources - --pretrained pde-transformer,dpot     # operator-only: external-target checkpoints, never mounted for agents
```

```bash
export TASK_PRIVATE_DIR=/absolute/path/on/large-disk/pde-surrogate
pip install huggingface-hub==0.36.2 h5py==3.16.0   # outside Harbor; the pins the images use

# an agent host, in a separate directory: the release's train/ subtree plus the upstream training pools (138 GB).
# --pools restricts every source to the files the split's train/valid pools reference and accepts the
# truncated heads the first step fetched (a verifier host, without --pools, refuses those as not-upstream)
python3 pde-surrogate/environment/data/materialize.py "$TASK_DATA_DIR" --from-release --subtree train
python3 pde-surrogate/environment/data/materialize.py "$TASK_DATA_DIR" --pools pde-surrogate/environment/data/split.json --sources -

# smoke slice: reduced corpus (9.3 GB) and the smallest Well dataset (7.9 GB)
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" \
  --sources ape2d,well-turbulent_radiative_layer_2D

# full pretraining corpus (78.7 GB)
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" \
  --sources ape2d-full

# large Well datasets: restrict to named files or splits
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" \
  --sources well-rayleigh_benard --splits valid,test
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" \
  --sources ape2d-full --plan          # print the file list, download nothing
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" \
  --sources ape2d --verify-only        # offline re-check of existing files
```

`materialize.py` downloads each file individually at the pinned revision with
`hf_hub_download`, checks its SHA-256 against the upstream LFS digest, opens it
and compares the trajectory count with the count the lock inferred from the
byte size, then writes `materialize-receipt.json`. It is safe to rerun; existing
files are verified, not re-downloaded. Any mismatch fails the run. `--pools
split.json` keeps the held-out files off an agent host (the corpus's `train`
files carry the held-out tails, so `--splits` cannot); the development
container's entrypoint (`environment/data/check_agent_mount.py`) refuses a corpus
file that still carries a held-out tail.

Prepared layout:

```text
$TASK_PRIVATE_DIR/
  materialize-receipt.json
  ape2d/<name>/<name>.hdf5, <name>/meta_all.json
  ape2d-full/<name>.hdf5, <name>.json
  well-<dataset>/data/{train,valid,test}/<dataset>_<parameters>.hdf5
```

## Verified so far

- 2026-09-16, every locally downloaded file of the five sources: 282 files,
  182.4 GB, 13,212 trajectories; every SHA-256 matched the upstream digest and every
  trajectory count read from the file matched the lock
  (`../../docs/validation/materialize-receipt-2026-09-16.json`).
- The reduced corpus is **not** a subset of the full corpus: same per-simulation
  constants, different arrays (`docs/validation/reduced_vs_full.json`); the smoke
  split is plumbing only.
- Not downloaded: `rayleigh_benard` and `shear_flow` (optional sources, not in the split); the
  remaining PDEArena validation shards.
- 2026-09-22: every file of the full corpus fingerprinted by content; `hyp.hdf5` is a copy of
  `diff.hdf5` and is excluded from the split (`AUDIT.md` §7). Release `pde-surrogate-split-v2`
  built with a reader-level check on every member, verified anonymously from the Hub and
  content-checked for leakage (`docs/validation/release_leakage.json`).

## The published release

The data is published as one pinned release and this is the fastest way to get it:

```bash
python3 materialize.py DEST --from-release --subtree train    # an agent host
python3 materialize.py DEST --from-release                    # a verifier host: both subtrees
python3 materialize.py DEST --from-release --plan             # list, download nothing
```

Members are stored as `<subtree>/<source>/<path>` but land as `<source>/<path>`, the layout a
direct materialization produces, so nothing downstream has to know subtrees exist. For the nine
files the split cuts in two, a verifier host fetches both halves and `materialize.py` merges
them back into the whole upstream file — same trajectory count, same numbering, upstream's own
`norm_*` statistics, which only the `eval/` half preserves. An agent host asking for
`--subtree train` gets the truncated head alone, which is the point of the separation.
Every member keeps the `sims` group attributes the readers require, and `materialize.py`
opens each corpus member it fetches or merges and refuses one it cannot read.

| | |
| --- | --- |
| repository | [deqing/ai4sci-pde-surrogate-data](https://huggingface.co/datasets/deqing/ai4sci-pde-surrogate-data), public and ungated |
| pinned revision | `f99c579ead823535878e5022cdddce968956c1a2` |
| release | `pde-surrogate-split-v2`, kind `derived-only` (v1 @ `5bcba793` is superseded: its members lacked the group attributes and it carried the duplicated `hyp` family) |
| payload | 18 members, 39.32 GB (`train/` 32.56 GB, `eval/` 6.76 GB) |
| pin | `download-source.json` — repo id, revision, manifest SHA-256 |

The release is **derived-only**: it carries the 39.32 GB this split creates and nothing
else. The other 129 GB a verifier host uses (106 GB for an agent host) is byte-identical
to files already pinned and public at the MIT and CC-BY-4.0 upstreams in the table above,
and `materialize.py` fetches it there against the same SHA-256 recorded in
`sources.lock.json`. Mirroring those bytes would duplicate other people's data to add nothing.

Every member carries a SHA-256 in `payload-files.jsonl`, and the manifest that lists them
carries its own, pinned in `download-source.json`. A consumer verifies the manifest against
that pin, every file against the manifest, and trusts nothing else: an already-correct file
is left alone, and a file whose bytes disagree is reported rather than overwritten.
`PUBLICATION.md` is the record of what was published and what was deliberately not.

### Rebuilding or republishing (maintainers)

```bash
python3 build_release.py --data "$TASK_PRIVATE_DIR" --out /large-disk/pde-release   # release id defaults to the published one; every member is opened with pde_io
python3 ../../docs/validation/check_release_leakage.py \
    --release /large-disk/pde-release --data "$TASK_PRIVATE_DIR" \
    --out ../../docs/validation/release_leakage.json          # must print "clean"
python3 publish_release.py --release /large-disk/pde-release \
    --repo deqing/ai4sci-pde-surrogate-data --dry-run    # then without --dry-run
```

`--mirror` includes the upstream-identical members too, for an account with the storage
plan for 185 GB; Hugging Face's free public storage is best-effort rather than a quota.
`publish_release.py` writes `download-source.json` after the upload, pushing the manifest
last so a half-finished upload never looks complete.

The release has two subtrees, and which one a host fetches is what separates an agent
environment from a verifier environment:

| subtree | holds | mounted for |
| --- | --- | --- |
| `train/` | the training and validation pools. The nine corpus files that also carry held-out trajectories appear here truncated at the end of the validation range, with `norm_*` recomputed over the kept trajectories so the public head carries no aggregate information about the tail | the agent |
| `eval/` | the held-out files, the held-out trajectory tails under their original numbering, and any metadata describing the held-out set — PDEArena's `pde_test_seed_*.yaml` are the generator seeds for the test trajectories | the verifier only |

Together they reconstruct the upstream files exactly and neither alone gives an agent the
answers. Publishing the evaluation half is consistent with what `instruction.md` already
says: "All upstream data are public, so this is a benchmark-visible evaluation, not a secret
test set." Nothing under `pretrained/` is ever included.

`check_release_leakage.py` is the check that matters. Truncation is an argument about
indices; the checker tests it against content, hashing every trajectory in `train/` and every
held-out trajectory in the source and requiring the two sets to be disjoint. Bytes matching
a manifest is not enough either: the first release passed every hash check while its members
were unreadable, so the builder and the materializer now open every member with the task's
own reader (`tests/test_release.py`).

Two deliberate differences from foldbench's release, which solved a different problem. It
sharded into tars because it had 186k small feature files and a flat folder on the Hub is
unusable; this one is members of 0.1–9.4 GB, where a tar layer would only defeat per-file
resumption and range requests, so the guarantee is a SHA-256 per member instead. And its
payload is one undivided suite, whereas here the agent and verifier halves have to stay
apart, which is what the two subtrees are for.

Licensing permits the republication outright: `ape2d-full` and PDEArena are MIT, both Well
datasets are CC-BY-4.0, and all four allow redistribution with attribution and the licence
text. `tests/test_release.py` asserts that the unlicensed baseline families (`UNetConvNext`,
`FNO`) and the non-commercial ones (`Poseidon`) are not sources and can never enter a
release.
