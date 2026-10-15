#!/usr/bin/env python3
"""Predict the test split once with the best-validation checkpoint of a fine-tune.

Runs the official unimol2 inference path (task.valid_step with test-time conformer averaging over conf_size
conformers) and writes OUT (--predictions): one row per SMILES with predict_0..predict_{k-1}, the mean over
conformers and duplicate SMILES, exactly the aggregation of the official reduce_metrics (agg_mae). The labels in
the test LMDB are not read here (the verifier's test inputs hold placeholders). Arguments are those of
unimol2/infer.py plus --predictions.
"""
import csv
import json
import logging
import sys

import torch
from unicore import checkpoint_utils, options, tasks, utils

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level="INFO", stream=sys.stdout)


def main():
    i = sys.argv.index("--predictions"); out = sys.argv[i + 1]; del sys.argv[i:i + 2]
    parser = options.get_validation_parser(); options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    utils.import_user_module(args)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args); model = task.build_model(args)
    model.load_state_dict(state["model"], strict=True)
    if args.fp16:
        model.half()
    model.cuda().eval()
    loss = task.build_loss(args); loss.eval()
    task.load_dataset("test", combine=False, epoch=1)
    itr = task.get_batch_iterator(dataset=task.dataset("test"), batch_size=args.batch_size, ignore_invalid_inputs=False,
                                  seed=args.seed, num_workers=args.num_workers).next_epoch_itr(shuffle=False)
    sums, counts = {}, {}
    with torch.no_grad():
        for sample in itr:
            if len(sample) == 0:
                continue
            _, _, log = task.valid_step(utils.move_to_cuda(sample), model, loss, test=True)
            pred = log["predict"].float().cpu().tolist()
            for smi, p in zip(log["smi_name"], pred):
                acc = sums.setdefault(smi, [0.0] * len(p))
                for k, v in enumerate(p):
                    acc[k] += v
                counts[smi] = counts.get(smi, 0) + 1
    k = args.num_classes
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smi"] + [f"predict_{j}" for j in range(k)])
        for smi in sorted(sums):
            w.writerow([smi] + [repr(v / counts[smi]) for v in sums[smi]])
    best_epoch = state.get("extra_state", {}).get("train_iterator", {}).get("epoch", 1) - 1
    json.dump({"best_valid_epoch": best_epoch, "molecules": len(sums)}, open(out + ".json", "w"))
    print(f"wrote {len(sums)} molecules to {out} (best validation epoch {best_epoch})")


if __name__ == "__main__":
    main()
