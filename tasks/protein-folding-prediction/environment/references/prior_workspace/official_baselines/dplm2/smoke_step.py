#!/usr/bin/env python3
"""Exercise the upstream DPLM-2 loss graph for one non-comparable GPU step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LORA_TARGET = "(esm.encoder.layer.[0-9]*.attention.(self.query|self.key|self.value|output.dense).*|esm.encoder.layer.[0-9]*.(intermediate|output).dense.*)"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--records",
        type=Path,
        help="optional unified-data token shard; first four records are used",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    import torch
    from omegaconf import OmegaConf
    from byprot.datamodules.dataset.tokenized_protein import DPLM2Collater
    from byprot.models.dplm2.dplm2 import MultimodalDiffusionProteinLanguageModel
    from byprot.modules.cross_entropy import StructAARDMCrossEntropyLoss

    if not torch.cuda.is_available():
        raise RuntimeError("this smoke must run on an explicitly selected GPU")
    torch.manual_seed(42)
    config = OmegaConf.create(
        {
            "num_diffusion_timesteps": 500,
            "gradient_ckpt": False,
            "training_stage": "continue_train_from_dplm2",
            "single_modality_ratio": 0.25,
            "folding_loss_ratio": 0.25,
            "inverse_folding_loss_ratio": 0.25,
            "joint_loss_ratio": 0.25,
            "independent_loss_ratio": 0.0,
            "net": {
                "arch_type": "esm",
                "name": str(args.config),
                "dropout": 0.1,
                "pretrain": False,
                "pretrained_model_name_or_path": "",
            },
            "tokenizer": {"vocab_file": str(args.config), "vocab_size": 8229},
            "struct_tokenizer": {
                "enable": False,
                "exp_path": "airkingbd/struct_tokenizer",
            },
            "lora": {
                "enable": True,
                "lora_rank": 16,
                "lora_dropout": 0.1,
                "lora_target_module": LORA_TARGET,
                "modules_to_save": "lm_head,esm.embeddings",
            },
            "self_mixup": {"enable": False, "with_original_loss": False},
        }
    )
    model = MultimodalDiffusionProteinLanguageModel(config).cuda().train()
    collater = DPLM2Collater(model.tokenizer)
    if args.records:
        source_rows = [
            json.loads(line) for line in args.records.read_text().splitlines()[:4]
        ]
        if len(source_rows) != 4:
            raise ValueError("--records must contain at least four examples")
        raw = [
            {
                "aatype_tokens": f"<cls_aa>{row['aa_seq']}<eos_aa>",
                "struct_tokens": (
                    "<cls_struct>"
                    + "".join(row["struct_seq"].split(","))
                    + "<eos_struct>"
                ),
                "length": row["length"] + 2,
            }
            for row in source_rows
        ]
        residues = [row["length"] for row in source_rows]
        data_source = "foldingtrain-unified-split-v1"
    else:
        amino_acids = "LAGVSERTIDPKQNFY"
        structure = "".join(f"{index:04d}" for index in range(len(amino_acids)))
        raw = [
            {
                "aatype_tokens": f"<cls_aa>{amino_acids}<eos_aa>",
                "struct_tokens": f"<cls_struct>{structure}<eos_struct>",
                "length": len(amino_acids) + 2,
            }
            for _ in range(4)
        ]
        residues = [len(amino_acids)] * 4
        data_source = "synthetic"
    batch = {
        group: {key: value.cuda() for key, value in values.items()}
        for group, values in collater(raw).items()
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )
    criterion = StructAARDMCrossEntropyLoss(label_smoothing=0.0, ignore_index=1)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, targets, masks, weights = model.compute_loss(batch, weighting="linear")
        loss, logging = criterion(
            logits,
            targets,
            masks,
            weights,
            watch_t1_t2_loss=False,
            cal_constant_loss=True,
        )
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite loss: {loss}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    report = {
        "schema_version": "official-dplm2-loss-graph-smoke-v1",
        "quality_comparable": False,
        "device": torch.cuda.get_device_name(0),
        "data_source": data_source,
        "batch_size": 4,
        "residues_per_example": residues,
        "loss": float(loss.detach().cpu()),
        "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
        "total_parameters_with_lora_wrappers": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters_with_lora": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "official_top_level_class": f"{type(model).__module__}.{type(model).__name__}",
        "criterion_class": f"{type(criterion).__module__}.{type(criterion).__name__}",
        "logged_metrics": sorted(logging),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
