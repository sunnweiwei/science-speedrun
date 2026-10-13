# Materializing the timeseries-forecasting data (controller and maintainers)

This file is not copied into the agent's workspace. One materialized directory serves both
sides. Set `TASK_DATA_DIR` and `TASK_PRIVATE_DIR` to it. The development compose mounts
`gifteval_pretrain/`, `valid/` and the receipt under `/workspace/data`, and the verifier compose
mounts `gifteval/`.

```
python3 materialize.py /large-disk/timeseries                       # gifteval + the 95 corpus directories (275 GB)
python3 materialize.py /large-disk/timeseries --tier fast            # gifteval + the 64 small directories (1.25 GB), smoke runs
python3 materialize.py /large-disk/timeseries --sources gifteval     # verifier-side data only
python3 materialize.py /large-disk/timeseries --sources leaderboard  # reference CSVs
python3 make_split.py --data /large-disk/timeseries                  # regenerate split.json / split_smoke.json (frozen copies committed)
python3 ../evaluation/make_cases.py --data /large-disk/timeseries --manifest split.json --split valid \
    --input /large-disk/timeseries/valid/input --truth /large-disk/timeseries/valid/truth   # the agent's validation cases
```

The agent container mounts `gifteval_pretrain/` and `valid/` read-only under
`/workspace/data`; `gifteval/` (which contains the test region) is mounted only in the
verifier. `entrypoint.sh` warns if it finds `/workspace/data/gifteval`.

