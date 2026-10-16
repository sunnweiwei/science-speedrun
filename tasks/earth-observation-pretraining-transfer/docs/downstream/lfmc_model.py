"""Official LFMC UNet and masked MSE with exact local encoder initialization."""

import torch
from rslearn.models.component import FeatureMaps
from rslearn.models.unet import UNetDecoder
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.tasks.per_pixel_regression import PerPixelRegressionHead
from portable_encoder import PortableAWFEncoder
from masked_spatial import spatial_features


class LFMCModel(torch.nn.Module):
    def __init__(self, package):
        super().__init__()
        loaded = PortableAWFEncoder(package)
        self.encoder = loaded.model
        self.encoder_sha256 = loaded.encoder_sha256
        self.decoder = UNetDecoder(
            in_channels=[[4, loaded.embedding_size]],
            out_channels=1,
            conv_layers_per_resolution=2,
            num_channels={8: 512, 4: 512, 2: 256, 1: 128},
        )
        self.head = PerPixelRegressionHead()

    def forward(self, pixels, valid, target=None, target_valid=None):
        feature = spatial_features(self.encoder, pixels, valid)
        context = ModelContext(inputs=[], metadatas=[])
        logits = self.decoder(FeatureMaps([feature]), context)
        targets = None
        if target is not None:
            targets = [
                {
                    "values": RasterImage(y[None, None]),
                    "valid": RasterImage(v[None, None]),
                }
                for y, v in zip(target, target_valid)
            ]
        return self.head(logits, context, targets)


def backward_microbatches(model, batch, device, microbatch=4):
    """Exactly weight valid pixels across one optimizer batch (default size32).

    No optimizer step here. Caller zeros gradients once, calls this, then steps
    once. This preserves official batch-level masked MSE for unequal mask counts.
    """
    if microbatch <= 0:
        raise ValueError("Microbatch must be positive")
    count = int(batch["target_valid"].sum())
    total_loss = 0.0
    for start in range(0, len(batch["ids"]), microbatch):
        end = start + microbatch
        inputs = {k: v[start:end].to(device) for k, v in batch["pixels"].items()}
        valid = {k: v[start:end].to(device) for k, v in batch["valid"].items()}
        target = batch["target"][start:end].to(device)
        mask = batch["target_valid"][start:end].to(device)
        result = model(inputs, valid, target, mask)
        weight = float(mask.sum()) / count if count else 1.0
        loss = result.loss_dict["regress"] * weight
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite LFMC loss")
        loss.backward()
        total_loss += float(loss.detach())
    return total_loss, count
