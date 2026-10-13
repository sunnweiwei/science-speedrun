# Development evaluation

Self-service replay on the validation trajectories with the same grader the
verifier uses. Never build cases from the manifest's held-out `cases`; they are
replay-only.

```bash
E=/workspace/evaluation
python3 $E/make_valid_cases.py --manifest /workspace/data/split.json --out /workspace/research/dev/valid_cases.json
python3 $E/make_cases.py --data /workspace/data --manifest /workspace/research/dev/valid_cases.json --split valid \
  --input /workspace/research/dev/in --truth /workspace/research/dev/truth --context-frames 4
bash /workspace/submission/infer.sh /workspace/research/dev/in /workspace/research/dev/out
python3 $E/score.py --input /workspace/research/dev/in --truth /workspace/research/dev/truth --output /workspace/research/dev/out --report /workspace/research/dev/report.json
```

`make_valid_cases.py` draws one window per validation trajectory at a
deterministic offset (585 cases for the canonical split, whose pools exclude the duplicated `hyp` family; `--per-trajectory`
adds more). `score.py` reports per-case, per-dataset and per-group VRMSE, the
joint-channel nRMSE of the PDE-Transformer paper for context, The Well's
rollout windows (steps 1–5, 6–12, 13–30) where K allows, and the reward. Failed
cases are listed with reasons and stay in the denominator. Validation numbers
are for model selection; they are not the held-out score.
