#!/usr/bin/env python3
"""Check an encoder package against contract.md before any fine-tune (runs the package's code, so the verifier
runs it as the unprivileged user).

* encoder.json: {"arch": registered Uni-Core architecture (default unimol2_84M), "user_dir": optional relative
  directory with an __init__.py that registers it}; checkpoint.pt: a torch checkpoint with a "model" state dict.
* The fine-tune model (task mol_finetune, one regression head) is built from encoder.json. Its learned
  parameters, excluding the classification heads, must number at most MAX_PARAMS (90 M).
* Every parameter and buffer of that model outside the classification heads must be in checkpoint["model"] with
  the same shape (Uni-Core loads fine-tune weights with strict=False, so a mismatch would otherwise silently
  fine-tune a partly random model). Extra checkpoint entries (pretraining heads) are allowed.
usage: check_encoder.py PACKAGE DATA OUT_JSON   (DATA: any suite directory with task_metainfo.json)
"""
import json
import os
import sys
from pathlib import Path

MAX_PARAMS = 90_000_000
HEADS = "classification_heads."


def main():
    pkg, data, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    here = Path(__file__).resolve().parent
    spec = json.loads((pkg / "encoder.json").read_text())
    if not isinstance(spec, dict) or set(spec) - {"arch", "user_dir", "description"}:
        raise SystemExit("encoder.json must be an object with keys among arch, user_dir, description")
    arch = spec.get("arch", "unimol2_84M")
    if spec.get("user_dir"):
        ud = (pkg / spec["user_dir"]).resolve()
        if pkg not in ud.parents or not (ud / "__init__.py").is_file():
            raise SystemExit("user_dir must be a directory inside the package with an __init__.py")
        os.environ["UNIMOL2X_SUBMISSION_USER_DIR"] = str(ud)
    os.environ["UNIMOL2X_METAINFO"] = str(data / "task_metainfo.json")
    import torch
    from unicore import options, tasks, utils
    sys.argv = [sys.argv[0], str(data), "--user-dir", str(here / "unimol2x"), "--task", "mol_finetune",
                "--loss", "finetune_smooth_mae", "--arch", arch, "--task-name", "qm9dft_v2_alpha",
                "--classification-head-name", "check", "--num-classes", "1", "--reg"]
    parser = options.get_training_parser(); args = options.parse_args_and_arch(parser)
    utils.import_user_module(args)
    model = tasks.setup_task(args).build_model(args)
    n_params = sum(p.numel() for k, p in model.named_parameters() if not k.startswith(HEADS))
    state = torch.load(pkg / "checkpoint.pt", map_location="cpu", weights_only=False)
    weights = state.get("model") if isinstance(state, dict) else None
    if not isinstance(weights, dict):
        raise SystemExit('checkpoint.pt must be a dict with a "model" state dict')
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items() if not k.startswith(HEADS)}
    missing = sorted(k for k in expected if k not in weights)
    shape = sorted(k for k in expected if k in weights and tuple(weights[k].shape) != expected[k])
    report = {"arch": arch, "user_dir": spec.get("user_dir"), "encoder_parameters": n_params,
              "max_parameters": MAX_PARAMS, "missing_keys": missing[:20], "n_missing": len(missing),
              "shape_mismatch": shape[:20], "n_shape_mismatch": len(shape),
              "unused_checkpoint_keys": len([k for k in weights if k not in expected])}
    report["ok"] = n_params <= MAX_PARAMS and not missing and not shape
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    if not report["ok"]:
        raise SystemExit("encoder check failed: " + ("too many parameters" if n_params > MAX_PARAMS else
                                                     f"{len(missing)} missing / {len(shape)} mis-shaped weights"))


if __name__ == "__main__":
    main()
