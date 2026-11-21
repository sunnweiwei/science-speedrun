# Executable monomer starter

`infer.py` is the shared entry point for packaged reference policies. Used alone, it runs the released medium stochastic sampler on the current monomer request.

From the development workspace, publish just the standalone program and check it on a public request:

```bash
cd /workspace
candidate=$(mktemp -d /workspace/research/program.XXXXXX)
cp baseline/program/infer.py "$candidate/"
bash baseline/publish.sh "$candidate"
run=$(mktemp -d /workspace/research/program-check.XXXXXX)
python evaluation/run.py --length 119 --slots 2 --reference native-low --work "$run/work" --output "$run/score.json"
```

For the exact archived stochastic TSR policy package, follow the [stochastic TSR baseline](../stochastic-tsr/README.md). [Standalone program measurements](RESULTS.md)
