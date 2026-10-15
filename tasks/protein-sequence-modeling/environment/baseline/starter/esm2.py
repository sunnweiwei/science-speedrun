#!/usr/bin/env python3
"""An ESM-2 architecture, trained from scratch, as the reference baseline.

Follows Lin et al. (Science 2023) / facebookresearch/esm in the ways that
matter for a like-for-like architecture claim:

  * rotary position embeddings, not learned positional embeddings -- this is
    the defining architectural change from ESM-1b to ESM-2
  * pre-LayerNorm blocks, GELU feed-forward at 4x expansion
  * final LayerNorm before the LM head, and the LM head's projection weight
    tied to the token embedding
  * the ESM alphabet's special tokens (<cls>/<pad>/<eos>/<mask>) and the same
    20 standard + rare amino-acid symbols

The published size ladder is reproduced in CONFIGS so the baseline can be named
by its ESM-2 counterpart rather than by an arbitrary width.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ESM-2's alphabet, in its canonical order.
SPECIALS = ["<cls>", "<pad>", "<eos>", "<unk>"]
RESIDUES = list("LAGVSERTIDPKQNFYMHWCXBUZO")
EXTRA = ["<null_1>", "<mask>"]
ALPHABET = SPECIALS + RESIDUES + EXTRA
TOKEN = {symbol: index for index, symbol in enumerate(ALPHABET)}
CLS, PAD, EOS, UNK = TOKEN["<cls>"], TOKEN["<pad>"], TOKEN["<eos>"], TOKEN["<unk>"]
MASK = TOKEN["<mask>"]
VOCAB_SIZE = len(ALPHABET)
FIRST_RESIDUE, LAST_RESIDUE = TOKEN["L"], TOKEN["O"]

# name -> (layers, embed_dim, attention_heads), from the ESM-2 release table.
CONFIGS = {
    "esm2_t6_8M":    (6, 320, 20),
    "esm2_t12_35M":  (12, 480, 20),
    "esm2_t30_150M": (30, 640, 20),
    "esm2_t33_650M": (33, 1280, 20),
}


def encode(sequence, max_len=1024):
    body = [TOKEN.get(c, UNK) for c in sequence[:max_len - 2]]
    return [CLS] + body + [EOS]


def pad_batch(sequences, device):
    width = max(len(s) for s in sequences)
    out = torch.full((len(sequences), width), PAD, dtype=torch.long)
    for row, ids in enumerate(sequences):
        out[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out.to(device, non_blocking=True)


class Rotary(nn.Module):
    """Rotary position embedding, as ESM-2 applies it to q and k."""

    def __init__(self, dim, base=10000):
        super().__init__()
        inverse = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inverse", inverse, persistent=False)
        self._cached_length = 0

    def _tables(self, length, device, dtype):
        if length != self._cached_length or self._cos.device != device:
            position = torch.arange(length, device=device, dtype=torch.float)
            angles = torch.outer(position, self.inverse.to(device))
            table = torch.cat([angles, angles], dim=-1)
            self._cos = table.cos()[None, None]
            self._sin = table.sin()[None, None]
            self._cached_length = length
        return self._cos.to(dtype), self._sin.to(dtype)

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q, k):
        cos, sin = self._tables(q.shape[-2], q.device, q.dtype)
        return (q * cos + self._rotate_half(q) * sin,
                k * cos + self._rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x, pad_mask):
        batch, length, _ = x.shape
        shape = (batch, length, self.heads, self.head_dim)
        q = self.q(x).view(shape).transpose(1, 2)
        k = self.k(x).view(shape).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        q, k = self.rotary(q, k)
        attention_mask = None
        if pad_mask is not None:
            attention_mask = (~pad_mask)[:, None, None, :]
        hidden = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        hidden = hidden.transpose(1, 2).reshape(batch, length, -1)
        return self.out(hidden)


class Block(nn.Module):
    def __init__(self, dim, heads, ffn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))

    def forward(self, x, pad_mask):
        x = x + self.attention(self.norm1(x), pad_mask)
        return x + self.ffn(self.norm2(x))


class Encoder(nn.Module):
    """The shared backbone. Every task reads from this one parameter set."""

    def __init__(self, layers=12, dim=480, heads=20):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD)
        self.blocks = nn.ModuleList([Block(dim, heads, dim * 4) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens):
        pad_mask = tokens.eq(PAD)
        hidden = self.embed(tokens)
        for block in self.blocks:
            hidden = block(hidden, pad_mask)
        return self.norm(hidden), pad_mask

    def lm_logits(self, hidden):
        # Weight tying, as in ESM-2's RobertaLMHead.
        return F.linear(hidden, self.embed.weight) + self.lm_bias

    def pool(self, tokens):
        hidden, pad_mask = self.forward(tokens)
        keep = (~pad_mask).to(hidden.dtype).unsqueeze(-1)
        return (hidden * keep).sum(1) / keep.sum(1).clamp(min=1)


class Model(nn.Module):
    """ESM-2 backbone plus the eleven PEER task heads."""

    CLASSES = {"solubility": 2, "subcellular_localization": 10,
               "binary_localization": 2, "fold": 1195,
               "yeast_ppi": 2, "human_ppi": 2}
    REGRESSION = ("fluorescence", "stability", "beta_lactamase", "ppi_affinity")
    PAIR = {"yeast_ppi", "human_ppi", "ppi_affinity"}

    def __init__(self, layers=12, dim=480, heads=20):
        super().__init__()
        self.encoder = Encoder(layers=layers, dim=dim, heads=heads)
        self.heads = nn.ModuleDict()
        for task, n in self.CLASSES.items():
            width = dim * 2 if task in self.PAIR else dim
            self.heads[task] = nn.Sequential(
                nn.Linear(width, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, n))
        for task in self.REGRESSION:
            width = dim * 2 if task in self.PAIR else dim
            self.heads[task] = nn.Sequential(
                nn.Linear(width, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.heads["secondary_structure"] = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, 3))

    def represent(self, task, batch, device, max_len=1024):
        if task in self.PAIR:
            a = self.encoder.pool(pad_batch(
                [encode(x["sequence_a"], max_len) for x in batch], device))
            b = self.encoder.pool(pad_batch(
                [encode(x["sequence_b"], max_len) for x in batch], device))
            return torch.cat([a, b], dim=-1)
        return self.encoder.pool(pad_batch(
            [encode(x["sequence"], max_len) for x in batch], device))


def build(name="esm2_t12_35M"):
    layers, dim, heads = CONFIGS[name]
    model = Model(layers=layers, dim=dim, heads=heads)
    return model, {"name": name, "layers": layers, "dim": dim, "heads": heads}


if __name__ == "__main__":
    for name in CONFIGS:
        model, config = build(name)
        backbone = sum(p.numel() for p in model.encoder.parameters())
        total = sum(p.numel() for p in model.parameters())
        print(f"{name:16s} backbone {backbone/1e6:7.2f}M   with heads {total/1e6:7.2f}M")
