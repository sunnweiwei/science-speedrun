# Optimize RFD4-Proteina inference

Develop an executable inference program for frozen RFD4-Proteina. Each runtime request specifies the monomer input, number of outputs, seed and time budget. Choose the sampling, refinement, branching and selection algorithm in code. Maximize the number of structurally distinct folds that independently refold within 1.5 Å of their generated backbone. Compare quality and measured GPU time against the reproduced human baseline on the same requests.

## Workspace

- `/workspace/data` holds monomer inputs and frozen model assets.
- `/workspace/baseline` contains independent stochastic sampling, width-one branching and a two-wide, two-checkpoint beam, each with a policy and run record.
- `/workspace/references` links the original implementations, protocol and published comparisons.
- `/workspace/evaluation` runs development evaluation and contains the same scoring implementation used in verification.
- `/workspace/contract.md` describes the request/response interface, legacy policies, resource bounds and scoring.
- `/workspace/research` is writable scratch. Keep the current package in `/workspace/submission`.

The historical suite-level summaries are in `/workspace/references/overall/`. Its reference CSV identifies the fastest high-quality alternatives and the best end-to-end yield per suite. Use development draws to choose a policy. The controller owns the per-target archive and the final evaluation. The live monomer development and private runs share the same lengths.

## Rules

Use the supplied frozen model offline for inference on the current request. Produce new outputs from its input and seed. Model parameters, final refolding and clustering are fixed. The verifier supplies a per-request generation seed and keeps its private final AF3 seed separate. Use development measurements to select a program.

## Submission

Start from the reproduced stochastic TSR program. From `/workspace`, run `python evaluation/reference.py stochastic-tsr submission` with an empty submission directory. Edit `submission/infer.py` and include any helper modules it imports. The starter's policy adapter reproduces the human settings; your program may replace the entire implementation. Each program is called as `python infer.py --request FILE --output DIRECTORY`. Check a length outside the standard suite using `python evaluation/run.py --length 117 --slots 5 --work /tmp/rfd4-dev-117 --output /tmp/rfd4-dev-117.json` with fresh output paths.

## Continuous research

Use the reproduced development results and their logs. Check your own workflow with a short run. Keep experimenting and publishing improvements throughout the session.
