#!/usr/bin/env python3
"""Turn a fine-tuned Lightning checkpoint into a pde-surrogate-rollout/v1 package.

    python3 export_package.py --ckpt finetune/logs/ft-trl2d/checkpoint/epoch=000017.ckpt \
        --config finetune_trl2d.yaml --dataset-name well/turbulent_radiative_layer_2D --out pkg_ft_trl2d

The fine-tune runs on upstream's own trainer (`main.py`, `pdetransformer.core.separate_channels.Supervised`),
so the package keeps upstream's model and rollout: it stores the checkpoint's weights and the `model:` block
of the training config, rebuilds `Supervised` with `instantiate_from_config`, and calls its `predict()` on a
batch assembled from the task's case files. Nothing about the network or the rollout is re-implemented here;
only the batch dict that upstream's dataset would otherwise supply.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
PREDICT = '''#!/usr/bin/env python3
"""pde-surrogate-rollout/v1 inference for a fine-tuned PDE-Transformer (upstream Supervised.predict)."""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
from pdetransformer.data.metadata_remapping import all_pdes  # noqa: E402
from pdetransformer.utils import instantiate_from_config  # noqa: E402

SETUPS = json.loads((HERE / "setup.json").read_text())
if "dataset" in SETUPS:                       # single-dataset package: same file, one entry
    SETUPS = {SETUPS["dataset"]: SETUPS}
for _s in SETUPS.values():                    # the training pipeline's MeanStdNorm statistics
    _n = _s["normalization"]
    _s["_mean"] = np.array(_n["fields_mean"], dtype=np.float32).reshape(-1, 1, 1)
    _s["_std"] = np.array(_n["fields_std"], dtype=np.float32).reshape(-1, 1, 1)


def build_batch(SETUP, x, constants, device):
    """The fields upstream's dataset provides: (B, T, C, X, Y) data plus the physical metadata the model conditions on."""
    fields = SETUP["Fields"]
    NORM = SETUP["normalization"]
    pde_index = all_pdes.index(SETUP["PDE"].lower()) if SETUP["PDE"].lower() in all_pdes else 0
    # constants are normalized the same way the loader does (normalize_const: mean-std)
    c = (np.array(constants, dtype=np.float32) - np.array(NORM["const_mean"], dtype=np.float32)) / \
        np.maximum(np.array(NORM["const_std"], dtype=np.float32), 1e-10)
    const = torch.tensor([c], dtype=torch.float32, device=device)
    return {
        "data": torch.from_numpy(x[None]).to(device),
        "constants": const,
        "constants_norm": const,
        "time_step_stride": torch.tensor([[1.0]], dtype=torch.float32, device=device),   # loader's constant stride
        "physical_metadata": {
            "PDE": torch.tensor([pde_index], dtype=torch.long, device=device),
            "Constants": torch.tensor([SETUP["constant_classes"]], dtype=torch.long, device=device),
            # (batch, channel): a 1-D Fields tensor is read as one channel id per *sample*, which silently
            # sets the batch size to the channel count (upstream tasks.py get_channels)
            "Fields": torch.tensor([SETUP["field_classes"]], dtype=torch.long, device=device),
        },
    }


def main(input_dir, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load((HERE / "model_config.yaml").read_text())
    model = instantiate_from_config(cfg["model"]).to(device).eval()
    state = torch.load(HERE / "weights.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if missing:
        print(f"warning: {len(missing)} missing keys, first {missing[:3]}", file=sys.stderr)
    index = json.loads((Path(input_dir) / "index.json").read_text())
    (Path(output_dir) / "cases").mkdir(parents=True, exist_ok=True)
    for case in index["cases"]:
        SETUP = SETUPS.get(case["dataset"])
        keys = SETUP["constant_keys"] if SETUP else []
        with h5py.File(Path(input_dir) / "cases" / f"{case['case_id']}.h5", "r") as f:
            x = f["input"][()].astype(np.float32)
            constants = [float(json.loads(f.attrs["constants"]).get(k, 0.0)) for k in keys]
        K = int(case["n_steps_out"])
        if SETUP is None:
            # this model was not trained on this dataset, and has no calibrated statistics or metadata classes for
            # it, so it persists the last frame rather than guessing
            pred = np.repeat(x[-1][None], K, axis=0)
        else:
            FIELD_MEAN, FIELD_STD = SETUP["_mean"], SETUP["_std"]
            # upstream consumes `timesteps` frames (one input frame for the released sc-s recipe) and rolls one
            # step at a time; feed the last of the task's four context frames and take the K frames it appends
            t = int(getattr(model, "timesteps", 2))
            batch = build_batch(SETUP, (x[-t:] - FIELD_MEAN) / FIELD_STD, constants, device)
            with torch.no_grad():
                frames = model.predict(batch, device, num_frames=K + t - 1, output_type="tensor",
                                       return_dict=False, batch_dim=True)
            rolled = frames[0] if isinstance(frames, tuple) else frames      # (frames, reference)
            rolled = np.asarray(rolled.float().cpu() if torch.is_tensor(rolled) else rolled)
            pred = rolled[0, -K:] * FIELD_STD + FIELD_MEAN                    # (B, num_frames, C, X, Y) -> (K, C, X, Y)
        with h5py.File(Path(output_dir) / "cases" / f"{case['case_id']}.h5", "w") as f:
            f.create_dataset("prediction", data=pred.astype(np.float32))
    print(f"wrote {len(index['cases'])} predictions to {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''
INFER = '#!/usr/bin/env bash\nset -euo pipefail\nexec python3 "$(dirname "$(readlink -f "$0")")/predict.py" "$1" "$2"\n'

# upstream metadata for the datasets the fine-tunes cover (same table as convert_pools.py)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("convert_pools", HERE / "convert_pools.py")
convert_pools = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(convert_pools)
    SETUPS = convert_pools.SETUPS
except Exception:  # noqa: BLE001 - convert_pools imports the task's readers; fall back to the literal table
    SETUPS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-name", required=True,
                    help="dataset the fine-tune was trained on; comma-separated for a joint run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--field-classes", default="", help="comma-separated channel ids; default: 1..C from the setup")
    ap.add_argument("--pbdl-train", required=True,
                    help="the converted PBDL training file the fine-tune used (its cached normalization statistics); "
                         "comma-separated and matching --dataset-name when the model was trained on several pools")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)
    torch.save({"state_dict": state, "source_checkpoint": str(a.ckpt)}, out / "weights.pt")
    # the training config interpolates top-level values (${patch_size}, ${sample_size}); resolve them as upstream does
    from omegaconf import OmegaConf

    raw = OmegaConf.load(a.config)
    for key in ("data", "trainer", "lightning", "logger", "callbacks"):   # training-only blocks, some with env interpolations
        raw.pop(key, None)
    cfg = OmegaConf.to_container(raw, resolve=True)
    (out / "model_config.yaml").write_text(yaml.safe_dump({"model": cfg["model"]}))
    names = [s.strip() for s in a.dataset_name.split(",") if s.strip()]
    trains = [s.strip() for s in a.pbdl_train.split(",") if s.strip()]
    if len(trains) != len(names):
        raise SystemExit(f"--pbdl-train has {len(trains)} entries for {len(names)} datasets; they must correspond")
    # the classes the *training* pipeline encoded: upstream's fixed vocabularies, not positional 1..N.
    # Getting this wrong silently conditions the model on embeddings it never saw during the fine-tune.
    from pdetransformer.data.metadata_remapping import convert_constants, convert_fields

    import h5py

    blocks = {}
    for name, train in zip(names, trains):
        setup = SETUPS[name]
        fields = setup["Fields"]
        field_classes = ([int(x) for x in a.field_classes.split(",")] if a.field_classes and len(names) == 1
                         else [int(v) for v in convert_fields(fields).tolist()])
        with h5py.File(train, "r") as h:   # written by pbdl_dataloader/normalization.py when the loader first ran
            norm = {"fields_mean": [float(v) for v in h["norm_fields_sca_mean"][()].reshape(-1)],
                    "fields_std": [float(v) for v in h["norm_fields_sca_std"][()].reshape(-1)],
                    "const_mean": [float(v) for v in np.atleast_1d(h["norm_const_mean"][()]).reshape(-1)],
                    "const_std": [float(v) for v in np.atleast_1d(h["norm_const_std"][()]).reshape(-1)]}
        blocks[name] = {
            "dataset": name, "PDE": setup["PDE"], "Fields": fields, "field_classes": field_classes,
            "constant_keys": setup["_const_keys"], "constant_names": setup["Constants"],
            "constant_classes": [int(v) for v in convert_constants(setup["Constants"]).tolist()],
            "normalization": norm, "normalization_source": str(train),
        }
    # one dataset keeps the flat shape the first packages used; several are keyed by dataset name
    (out / "setup.json").write_text(json.dumps(blocks[names[0]] if len(names) == 1 else blocks, indent=1))
    (out / "predict.py").write_text(PREDICT)
    (out / "infer.sh").write_text(INFER); (out / "infer.sh").chmod(0o755)
    (out / "model_manifest.json").write_text(json.dumps({
        "contract": "pde-surrogate-rollout/v1",
        "name": "pde-transformer-sc-s-finetuned-" + ("joint" if len(names) > 1 else names[0].replace("/", "-")),
        "description": "Released PDE-Transformer sc-s fine-tuned with the paper's recipe on this task's training pool, "
                       "exported for replay; an external-target reference, not a submission",
        "parameters": int(sum(v.numel() for v in state.values() if hasattr(v, "numel"))),
        "context_frames": 1, "sources_used": [f"{n} (task training pool)" for n in names],
        "pretrained_assets": ["thuerey-group/pde-transformer@d5889aab16b1fe82e82940b612d54a8f0773e542 (sc-s)"],
    }, indent=1))
    shutil.copy(HERE / "README.md", out / "README.md")
    print(f"exported {a.ckpt} -> {out}")


if __name__ == "__main__":
    main()
