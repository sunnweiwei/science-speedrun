# Running everything without Harbor

Ordered commands a reviewer can run on a Linux box with one GPU, Python 3.11
or 3.12 and no Docker. Paths assume the repository root as the working
directory and `TASK_PRIVATE_DIR` on a large disk. Every step was run on
the contributor's A100 unless marked otherwise.

```bash
# 0. runtime (torch from the PyTorch index; the rest pinned, under the image's constraints).
#    The images use the cu126 build: cu130 needs driver >= 580, which H100 hosts on driver 570 lack.
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install -c pde-surrogate/environment/runtime/constraints.txt -r pde-surrogate/environment/runtime/requirements.txt

# 1. data. Verifier host: the 54 files the held-out cases reference (63 GB), from upstream or the release
export TASK_PRIVATE_DIR=/large-disk/pde-surrogate
bash pde-surrogate/environment/data/materialize.sh "$TASK_PRIVATE_DIR"                    # upstream, held-out files only
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" --from-release  # or: both release subtrees, 39 GB, merged back into whole files
#    Agent host (a different directory): the release's train/ subtree (33 GB) plus the upstream training pools (106 GB).
#    --pools fetches only the files the split's train/valid pools reference and accepts the release heads already there.
export TASK_DATA_DIR=/large-disk/pde-surrogate-agent
python3 pde-surrogate/environment/data/materialize.py "$TASK_DATA_DIR" --from-release --subtree train
python3 pde-surrogate/environment/data/materialize.py "$TASK_DATA_DIR" --pools pde-surrogate/environment/data/split.json --sources -
python3 pde-surrogate/environment/data/check_agent_mount.py "$TASK_DATA_DIR" pde-surrogate/environment/data/split.json   # what the dev entrypoint runs
#    Smoke slice (17 GB): the reduced corpus and the smallest Well dataset
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" --sources ape2d,well-turbulent_radiative_layer_2D
#    --plan prints any of the above without downloading; --verify-only re-hashes what is present

# 2. tests (data-free, ~25 s; pytest only - unittest discover finds nothing)
bash pde-surrogate/tests/run_unit_tests.sh

# 3. starter: train, publish, relocate, replay through the sealed verifier
cd pde-surrogate/environment/baseline/starter
python3 train.py --data "$TASK_PRIVATE_DIR" --manifest ../../data/split_smoke.json --out /tmp/run --steps 4000 --batch 8 --width 32 --max-windows-per-traj 12   # smoke, 227 s
python3 train.py --data "$TASK_DATA_DIR" --manifest ../../data/split.json --out /tmp/run --steps 30000 --batch 8 --width 32 --crop 128 --lr 1e-3 --seed 0   # the recorded run: 40 min
python3 publish.py /tmp/run /tmp/submission && cp -r /tmp/submission /tmp/elsewhere && cd -
python3 pde-surrogate/tests/main_verifier.py --submission /tmp/elsewhere --data "$TASK_PRIVATE_DIR" \
  --manifest pde-surrogate/environment/data/split_smoke.json --output /tmp/verifier-logs --work /tmp/verifier-work --timeout 3600
cat /tmp/verifier-logs/reward.json /tmp/verifier-logs/status.json
#    reward.json names the isolation mode: "bwrap" (user namespace), "uid" (root dropped to uid 10001), or "subprocess"
#    (only with PDE_SURROGATE_ALLOW_NO_ISOLATION=1, marked untrusted). --phase stage|infer|score splits the run for operators.

# 4. references (operator-only; the external targets use released weights agents never get)
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" --sources - --pretrained pde-transformer,dpot
R=pde-surrogate/docs/validation/released_models; B=pde-surrogate/environment/baseline/cnext_unet
python3 $R/pde_transformer_zero_shot/build.py --data "$TASK_PRIVATE_DIR" --source ape2d-full
PDET_CKPT_DIR="$TASK_PRIVATE_DIR/pretrained/pde-transformer" PDET_VARIANT=mc-s bash $R/pde_transformer_zero_shot/infer.sh INPUT_DIR OUTPUT_DIR
W=/large-disk/cnext   # CNextU-net work directory; inside the development image the defaults are /workspace/data, /workspace/research/cnext_unet and /opt/the_well (environment/baseline/cnext_unet/README.md)
DATA_DIR="$TASK_DATA_DIR" CNEXT_WORK=$W THE_WELL_SRC=/path/to/the_well@ad50de0e bash $B/train_dataset.sh turbulent_radiative_layer_2D turbulent_radiative_layer_2D 200   # ~12 h/dataset cap, The Well's recipe
python3 $B/convert_pdearena_to_well.py --data "$TASK_DATA_DIR" --data "$TASK_PRIVATE_DIR" --manifest pde-surrogate/environment/data/split.json --out $W/well-pdearena_navier_stokes_2d   # pools from the agent tree; held-out shards (verifier tree) only for the trainer's own test pass
python3 $B/package.py --run $W/runs/<run> --dataset-dir $W/well_base/turbulent_radiative_layer_2D --out /tmp/cnext --dataset-name well/turbulent_radiative_layer_2D --checkpoint best.pt
python3 $R/pde_transformer_finetune/convert_pools.py --data "$TASK_PRIVATE_DIR" --manifest pde-surrogate/environment/data/split.json --dataset well/turbulent_radiative_layer_2D --out /large-disk/pbdl
python3 $R/pde_transformer_finetune/make_lightning_ckpt.py --hf "$TASK_PRIVATE_DIR/pretrained/pde-transformer" --variant sc-s --out /large-disk/sc-s_init.ckpt
bash $R/pde_transformer_finetune/run_finetune_trl2d.sh            # upstream trainer, paper recipe

# 5. rebuilding the data release (maintainers)
python3 pde-surrogate/environment/data/build_release.py --data "$TASK_PRIVATE_DIR" --out /large-disk/pde-release   # validates every member with pde_io
python3 pde-surrogate/docs/validation/check_release_leakage.py --release /large-disk/pde-release --data "$TASK_PRIVATE_DIR" --out pde-surrogate/docs/validation/release_leakage.json
python3 pde-surrogate/environment/data/publish_release.py --release /large-disk/pde-release --repo deqing/ai4sci-pde-surrogate-data --dry-run

# 6. Docker. Harbor uses the compose overlays; both build with the task directory as context. The
#    integrator built and ran both on an 8x H100 host (2026-09-23): the verifier mounts the data under the
#    root-only /private, so a package running as the runner uid cannot regenerate the truth from it.
docker compose -f pde-surrogate/environment/docker-compose.yaml build      # TASK_DATA_DIR set
docker compose -f pde-surrogate/tests/docker-compose.yaml build            # TASK_PRIVATE_DIR set
```
