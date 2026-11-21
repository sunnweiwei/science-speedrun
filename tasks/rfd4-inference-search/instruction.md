# Continuously improve inference for RFD4-Proteina

## Task and goal

Develop an inference program for the supplied frozen RFD4-Proteina model. Given a runtime request for unconditional protein monomers, generate structurally distinct folds that AlphaFold 3 (AF3) can independently recover from their sequences. Repeated copies of one successful fold earn one point for that fold.

The reward is the mean clustered pass rate across the evaluated lengths. At each length, count the distinct Foldseek clusters with at least one backbone self-consistency RMSD ≤ 1.5 Å, then divide by all requested output slots. Missing and failed outputs stay in that denominator. Improve this score under the request budget and compare actual cost against a reproduced human baseline. The development report includes the paired quality difference and end-to-end GPU time, including final AF3.

## Workspace

- `/workspace/instruction.md` describes the research objective.
- `/workspace/contract.md` specifies the program interface, request types, resource limits and scoring.
- `/workspace/data/` contains the data guide and read-only monomer inputs, RFD4 weights, AF3 parameters and Foldseek executable.
- `/workspace/baseline/` contains the executable starter, stochastic TSR, two branching policies, and native Low and High presets, with public run records.
- `/workspace/references/` contains the upstream monomer example and published quality–throughput comparisons.
- `/workspace/evaluation/` contains the development evaluator, package builder and curve exporter.
- `/workspace/submission/` holds your current complete, runnable package.
- `/workspace/research/` is writable scratch for candidates, logs, measurements and derived data.

## Rules

Use the supplied assets offline. Keep the model parameters fixed and generate fresh outputs from each request's input and seed. You may change the sampling algorithm, search, branching, batching, stopping decisions and use of intermediate scores. Read the request at runtime so the same package can handle other supported lengths and output counts.

Use public development draws to select algorithms. Private evaluation data and seeds are off limits. Final AF3 refolding and structural clustering use the common evaluator.

## Submission

Publish a complete package that follows `/workspace/contract.md` early. Its entry point is `infer.py --request FILE --output DIRECTORY`. Bundle every helper module and runtime file your code needs besides the supplied assets and installed dependencies.

From `/workspace`, build and publish the reproduced stochastic TSR as a starting point.

```bash
candidate=$(mktemp -d /workspace/research/candidate.XXXXXX)
python evaluation/reference.py stochastic-tsr "$candidate"
bash baseline/publish.sh "$candidate"
```

Develop in the candidate directory. For a short check, score it on a public request using the development evaluator.

```bash
run=$(mktemp -d /workspace/research/check.XXXXXX)
python evaluation/run.py --submission "$candidate" --length 117 --slots 5 \
  --work "$run/work" --output "$run/score.json"
```

Whenever a candidate improves, publish its entire directory with `bash baseline/publish.sh "$candidate"`. The publisher stages and atomically exchanges the complete package at `/workspace/submission`.

## Continuous research

The baselines have already been reproduced. Each baseline's `RESULTS.md` and `reproduction/` contain its measured evidence. Check your own pipeline with a short run, then use development measurements to guide improvements. Compare quality and actual cost on matched requests and seeds, including cases where search spends more time for little quality gain.

There is no end point. Keep a working submission available, record what changed and what it achieved, and start another experiment after each improvement. Continue researching and publishing throughout the session without stopping to hand in a final answer.
