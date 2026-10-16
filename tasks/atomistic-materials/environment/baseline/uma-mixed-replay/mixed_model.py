"""Fine-tuning model whose OMat24 config name shares the pretrained "omat" dataset embedding.

FAIRChem 2.16's checkpoint overrides cannot add a dataset mapping to a backbone saved with a
dataset list, so the alias is set after loading. The embedding table is unchanged (no new
parameters), and the saved model config still lists only the pretrained datasets, so the exported
checkpoint loads with the stock FAIRChem predictor and task "omat".
"""
from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model


def initialize_mixed_model(checkpoint_location, overrides=None, heads=None, strict=True, aliases=None):
    model = initialize_finetuning_model(checkpoint_location, overrides, heads, strict)
    backbone = model.backbone
    mapping = dict(backbone.dataset_mapping)
    for name, target in dict(aliases or {}).items():
        if target not in mapping:
            raise ValueError(f"alias target {target} is not a pretrained dataset")
        mapping[name] = mapping[target]
    backbone.dataset_mapping = mapping
    backbone.dataset_embedding.dataset_mapping = mapping
    return model
