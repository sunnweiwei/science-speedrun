# Frozen data publication — 2026-09-22 (`pde-surrogate-split-v2`)

This is a storage and publication change, not a new data selection or a new
benchmark result. GitHub keeps the Harbor task, contract, starter, reference
tooling and evaluator. Hugging Face hosts the large data files.

## Location and immutable identity

- Public, ungated dataset:
  [deqing/ai4sci-pde-surrogate-data](https://huggingface.co/datasets/deqing/ai4sci-pde-surrogate-data)
- Pinned HF commit: `f99c579ead823535878e5022cdddce968956c1a2`
- Release: `pde-surrogate-split-v2` (`derived-only`)
- Manifest SHA-256: `9fe628b4eb46e87b18293f09c2383517d24a4590924e6214897594ea7e8eb251`
- Payload: 18 members, 39.32 GB (`train/` 9 files 32.56 GB, `eval/` 9 files 6.76 GB)

`download-source.json` (repo id, revision, manifest SHA-256) is the download
contract checked into Git; `release-manifest.json` and the per-file inventory
`payload-files.jsonl` live beside the payload on the Hub at that revision.

## What v2 changes over v1

`pde-surrogate-split-v1` (revision `5bcba793`, 2026-09-20) stays on the Hub as
history but must not be used:

- its members were written without the `sims` group attributes (`PDE`,
  `Fields`, `Fields Scheme`, `Dt`, ...). Every byte matched its manifest, and
  the task's own readers could not open a single file. The writers copied
  datasets but not attributes, and no check read a member back with the reader
  the verifier and the starter use. The builder now validates every member
  with `pde_io.read_window`, `materialize.py` refuses a corpus member it cannot
  read, and `tests/test_release.py` reads every fixture back the same way;
- it carried upstream's `hyp.hdf5`, which is a content copy of `diff.hdf5`
  (`AUDIT.md` §7). The split no longer uses that file, so the release cuts nine
  corpus files instead of ten.

## Published payload

This release is **derived-only**: it carries the 39.32 GB this benchmark's split
creates and nothing else. The other 129 GB a verifier host uses (106 GB for an
agent host) is byte-identical to files already pinned and public at MIT and
CC-BY-4.0 upstreams, and `materialize.py` fetches it from there against the same
SHA-256 recorded in `sources.lock.json`. Mirroring those bytes would duplicate
other people's data to add nothing, and Hugging Face's free public storage is
best-effort rather than a quota. `build_release.py --mirror` produces the full
copy for an account with the storage plan for it.

Two subtrees, and which one a host fetches is what separates an agent
environment from a verifier environment.

| subtree | meaning |
| --- | --- |
| `train/` | the training and validation pools; what an agent environment may mount |
| `eval/` | the held-out files and the held-out trajectory tails; what the verifier needs, never mounted for an agent |

Together they reconstruct the upstream files exactly; neither alone gives an
agent the answers. The nine corpus files that carry both appear in both subtrees,
split at the end of the validation range, with `norm_*` recomputed over the kept
trajectories in `train/` so the public head carries no aggregate information
about the tail.

Publishing the evaluation half is consistent with what the task already claims:
"All upstream data are public, so this is a benchmark-visible evaluation, not a
secret test set" (`instruction.md`). Anyone can download the MIT-licensed corpus
and read trajectory 500 of `burgers.hdf5`.

**Not published:** anything under `pretrained/`. Those checkpoints are pinned
downloads with their own licences, and two audited baseline families
(`UNetConvNext`, `FNO`) carry no licence at all upstream. `tests/test_release.py`
asserts they can never enter a release. PDEArena's `pde_test_seed_*.yaml`, the
generator seeds of the held-out trajectories, are metadata that would only ever
belong to `eval/`; a derived-only release carries no metadata at all.

## Licensing

The only republished bytes derive from `thuerey-group/pde-transformer-ape2d-full`
(MIT); the dataset card names the licence and the authors, and no relicensing is
claimed. The Well datasets (CC-BY-4.0) and PDEArena (MIT) are not republished.
See `docs/UPSTREAM_LICENSES.md`.

## Verified after upload

Checked against the Hub, anonymously (no token), rather than the uploader's own
report:

- repository public and ungated; `release-manifest.json` fetched at the pinned
  revision hashes to `9fe628b4eb46e87b…`, matching `download-source.json`; 24
  files at that revision (18 members, manifest, inventory, card, `.gitattributes`).
- `eval/ape2d-full/gs_delta.hdf5`, the smallest member, round-tripped: its
  SHA-256 matches the inventory and `pde_io.read_window` opens it (Gray-Scott,
  two fields, trajectories 80..99 under original numbering).
- `materialize.py DEST --from-release --subtree train --plan` lists the 9
  `train/` members, 32.56 GB.
- `docs/validation/release_leakage.json`: 580 held-out trajectories hashed against
  the `train/` subtree, 0 content collisions, 0 structure failures, 0 held-out
  files present in `train/`.

## Integrity and reproducibility checks

- `build_release.py` hashes every member while writing, records the upstream
  digest beside it, and opens every derived member with the task's reader; a
  member the reader cannot open fails the build.
- `docs/validation/check_release_leakage.py` tests truncation against content:
  every trajectory in `train/` and every held-out trajectory in the source are
  hashed and the two sets must be disjoint. It also fails a release that is
  missing files or that carries a held-out file in `train/`.
- `materialize.py --from-release` verifies the manifest against the pin, every
  file against the manifest, opens every corpus member, leaves an already-correct
  file alone, and reports rather than overwrites a file whose bytes disagree.
  Members land at `<dest>/<source>/<path>`, the layout the direct path produces
  and every reader assumes (`tests/test_release.py::test_from_release_lands_members_at_the_direct_layout`).

## Why this release is not tar-sharded

foldbench's release is 186k small feature files, where a flat folder on the Hub
is unusable and tar shards are the right answer. This one is 18 members of
0.3–7.9 GB, where a tar layer would only defeat per-file resumption and range
requests. The guarantee is the same — a SHA-256 per member, checked on download —
under a simpler layout, plus the reader-level check foldbench's `validate_release.py`
performs and v1 lacked.
