# Optimize RFD4-Proteina inference

Develop a search policy for a frozen flow model that generates unconditional protein monomers. At each of three lengths, a fixed budget can be spent on refinement, exploring sibling trajectories and scoring partial or completed trajectories. Maximize the number of structurally distinct folds that independently refold within 1.5 Å of their generated backbone. The score is the mean clustered pass rate across lengths.

## Workspace

- `/workspace/data` holds monomer inputs and frozen model assets.
- `/workspace/baseline` contains independent stochastic sampling, width-one branching and a two-wide, two-checkpoint beam, each with a policy and run record.
- `/workspace/references` links the original implementations, protocol and published comparisons.
- `/workspace/evaluation` runs development evaluation and contains the same scoring implementation used in verification.
- `/workspace/contract.md` describes the policy language, resource bounds and scoring.
- `/workspace/research` is writable scratch. Keep the current package in `/workspace/submission`.

The historical suite-level summaries are in `/workspace/references/overall/`. Its reference CSV identifies the fastest high-quality alternatives and the best end-to-end yield per suite. Use development draws to choose a policy. The controller owns the per-target archive and the final evaluation. The live monomer development and private runs share the same lengths.

## Rules

Use the supplied assets offline. Optimize the inference policy. Model parameters, the monomer problems, final refolding and clustering are fixed. Evaluation seeds are private. Work from development results only.

## Submission

Start by publishing the strongest locally measured monomer policy, stochastic TSR (82 distinct passing clusters in 150 public slots). From `/workspace` run `bash baseline/publish.sh baseline/stochastic-tsr`, then continue from that package and replace it atomically as performance improves.

## Continuous research

Use the reproduced development results and their logs. Check your own workflow with a short run. Keep experimenting and publishing improvements throughout the session.
