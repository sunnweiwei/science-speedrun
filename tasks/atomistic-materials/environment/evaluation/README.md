# Development evaluation

The verifier's own drivers and scorers. The verifier runs them on the held-out test split; run
them on the validation split under `/workspace/data/val/` with the same commands. The commands
for all eight suites are in `/workspace/contract.md` ("What the evaluator runs").

| file | what it does |
| --- | --- |
| `run_local.py` | runs the physical suites (discovery, phonons, elasticity) with your package, one worker per listed GPU |
| `score.py` | scores a `run_local.py` run; run it with the MP2020 runtime, `/opt/scoring-venv/bin/python` |
| `extra.py run --suite kappa\|mof\|amorphous\|cleavage\|neb` | runs one off-equilibrium suite; `--workers-per-gpu` adds workers per GPU |
| `score_extra.py` | scores an `extra.py` run against the references of the same split |
| `aggregate.py --metrics FILE` | turns flat validation metrics into the fixed score |

Every driver accepts any subset of a `val/<suite>/inputs` directory. A fixed subset of every suite is
the practical way to compare candidates; a full validation pass costs roughly two thirds of a test
replay (see the contract's limits).
