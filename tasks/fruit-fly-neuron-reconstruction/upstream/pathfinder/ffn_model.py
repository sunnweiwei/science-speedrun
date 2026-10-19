"""Independent FFN v1.5 architecture reconstruction, not author checkpoints.

20 residual blocks, 32 channels, per-voxel channel LayerNorm, additive seed-logit
updates. LayerNorm axes and initialization are explicit independent choices.
Optional training-set mean/std normalization is stored with the model state and
applied to the existing raw/255 input; checkpoints without it retain that input.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    def __init__(self,width):
        super().__init__();self.a=nn.Conv3d(width,width,3,padding=1);self.b=nn.Conv3d(width,width,3,padding=1)
        self.norm=nn.LayerNorm(width)
    def forward(self,x):
        x=x+self.b(F.relu(self.a(F.relu(x))))
        return self.norm(x.movedim(1,-1)).movedim(-1,1)


class FFN15(nn.Module):
    def __init__(self,width=32,blocks=20,activation_checkpointing=True,raw_normalization=None):
        super().__init__();self.stem=nn.Conv3d(2,width,3,padding=1)
        self.blocks=nn.ModuleList([Block(width) for _ in range(blocks)])
        self.out=nn.Conv3d(width,1,1);self.activation_checkpointing=activation_checkpointing
        norm=None if raw_normalization is None else torch.tensor(raw_normalization,dtype=torch.float32)
        self._validate_normalization(norm)
        self.register_buffer('raw_normalization',norm)
    @staticmethod
    def _validate_normalization(norm):
        if norm is not None and (norm.shape!=(2,) or not torch.isfinite(norm).all() or norm[1]<=0):
            raise ValueError('Raw normalization requires finite mean and positive std')
    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        value=state_dict.get(prefix+'raw_normalization')
        self._validate_normalization(value)
        if value is not None:
            if self.raw_normalization is None:
                self.raw_normalization=self.stem.weight.new_empty(2)
            elif not torch.equal(value.to(self.raw_normalization),self.raw_normalization):
                raise ValueError('Checkpoint normalization differs from configured normalization')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)
    def forward(self,raw,seed_logits):
        if self.raw_normalization is not None:
            raw=(raw-self.raw_normalization[0])/self.raw_normalization[1]
        x=self.stem(torch.cat((raw,seed_logits),dim=1))
        for block in self.blocks:
            x=checkpoint(block,x,use_reentrant=False) if self.training and self.activation_checkpointing else block(x)
        return seed_logits+self.out(F.relu(x))
