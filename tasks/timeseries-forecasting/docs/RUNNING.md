# Running the task

## Data (once, with network)

```
bash environment/data/materialize.sh /large-disk/timeseries                       # 43 GB: gifteval + the 81 allow-listed corpus directories
python3 environment/data/materialize.py /large-disk/timeseries --sources leaderboard
python3 environment/evaluation/make_cases.py --data /large-disk/timeseries --manifest environment/data/split.json \
    --split valid --input /large-disk/timeseries/valid/input --truth /large-disk/timeseries/valid/truth
```

One directory serves both containers: the development overlay
(`environment/docker-compose.yaml`) mounts only `gifteval_pretrain/`, `valid/`
and the receipt under `/data`, and `entrypoint.sh` refuses to start if
`/data/gifteval` is visible; the verifier overlay (`tests/docker-compose.yaml`)
mounts `gifteval/` alone. Both read `TASK_DATA_DIR`.

## Starter (the recorded run, 16 minutes on an A100-40GB after a 10-minute pack)

```
cd environment/baseline/starter
python3 data.py pack --data /large-disk/timeseries/gifteval_pretrain --out /large-disk/packed
python3 train.py --packed /large-disk/packed --out /tmp/run --steps 30000 --batch 64 --lr 5e-4 --context 1024 \
    --horizon 128 --patch 32 --synthetic-fraction 0.1 --valid /large-disk/timeseries/valid --seed 0
python3 publish.py --weights /tmp/run/best.pt --out /tmp/submission --sources-from /large-disk/packed --synthetic kernelsynth
```

## Agent runs (agent-runner)

```
agent-runner start --task timeseries-forecasting --run-dir /large-disk/agent-runs/ts-001 \
    --data-dir /large-disk/timeseries --data-env-var TASK_DATA_DIR \
    --agent <adapter> --model <provider/model> --network no-network --duration-hours 48 --gpus task
agent-runner evaluate --task timeseries-forecasting --run-dir /large-disk/agent-runs/ts-001 --snapshot <id>
```

`compute-profiles.toml`: `default` uses the task's GPU request; `smoke` is one GPU, 8 CPUs, 32 GiB for dry runs
with `environment/data/split_smoke.json` and the `fast` corpus tier.

## Verifying a package

Harbor: `tests/test.sh` in the verifier overlay (`TIMESERIES_SPLIT_MANIFEST=/tests/split_smoke.json` for a
plumbing run). Locally without Docker:

```
python3 tests/main_verifier.py --submission /path/to/package --data /large-disk/timeseries \
    --manifest environment/data/split.json --output /tmp/verify --work /tmp/verify-work --timeout 14400
```

Outputs `report.json` (per-config metrics, relatives, breakdowns), `results.csv` (leaderboard column names),
`reward.json` (reward, normalised score, isolation mode), `status.json`, `infer.log`. Inference runs in a
Bubblewrap namespace, or as uid 10001 when the verifier is root and the container cannot create one;
`--phase stage|infer|score` splits a run into phases for operators.

## Reference packages (maintainers)

```
python3 environment/data/materialize.py /large-disk/timeseries --sources - --pretrained timesfm-3.0,timesfm-2.5,chronos-2,chronos-bolt-base
bash environment/reference/<package>/infer.sh INPUT_DIR OUTPUT_DIR     # each README lists its venv and pins
```

## Isolation probe

```
python3 tests/check_local_isolation.py --data /large-disk/timeseries --out /tmp/isolation-probe
```

## Unit tests

`bash tests/run_unit_tests.sh` (pytest; no data or GPU needed; `unittest discover` finds nothing).

## Docker (not run by the contributor; the pde-surrogate twin images were built and run by the integrator on an H100 host)

```
TASK_DATA_DIR=/large-disk/timeseries docker compose -f environment/docker-compose.yaml build
TASK_DATA_DIR=/large-disk/timeseries docker compose -f tests/docker-compose.yaml build
```
