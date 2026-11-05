"""Author-style batch retrieval validation for BioLiP development pairs."""

from __future__ import annotations

import numpy as np
import torch
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from model import author_masked_logits


def masked_logits(logits: torch.Tensor, molecule_ids: list[str],
                  pocket_ids: list[str]) -> torch.Tensor:
    return author_masked_logits(logits, molecule_ids, pocket_ids)


def selection_improves(candidate: float, best: float) -> bool:
    """Maximize BEDROC; a tie keeps the earlier checkpoint."""
    return candidate > best


def retrieval_batch_metrics(logits: torch.Tensor, molecule_ids: list[str],
                            pocket_ids: list[str]) -> dict[str, float]:
    size = logits.shape[0]
    if size < 2:
        return {key: 0.0 for key in ("loss_sum", "bedroc_sum", "auc_sum",
                                      "top1_sum", "top3_sum", "queries")}
    masked = masked_logits(logits, molecule_ids, pocket_ids)
    target = torch.arange(size, device=logits.device)
    loss_sum = 0.5 * (
        torch.nn.functional.cross_entropy(masked.float(), target, reduction="sum")
        + torch.nn.functional.cross_entropy(masked.float().T, target, reduction="sum")
    )
    scores = masked.detach().float().cpu().numpy()
    bedroc = auc = top1 = top3 = 0.0
    for row in range(size):
        order = np.argsort(-scores[row], kind="stable")
        labels = np.zeros(size, dtype=np.float64)
        labels[row] = 1.0
        bedroc += float(CalcBEDROC(np.column_stack((scores[row, order], labels[order])), 1, 80.5))
        positive = scores[row, row]
        negatives = np.delete(scores[row], row)
        auc += float(((negatives < positive).sum() + 0.5 * (negatives == positive).sum())
                     / len(negatives))
        top1 += float(order[0] == row)
        top3 += float(row in order[:min(3, size)])
    return {"loss_sum": float(loss_sum), "bedroc_sum": bedroc,
            "auc_sum": auc, "top1_sum": top1, "top3_sum": top3,
            "queries": float(size)}


def aggregate_metrics(sums: dict[str, float]) -> dict[str, float]:
    count = sums["queries"]
    if count < 1:
        raise ValueError("validation contains no eligible retrieval queries")
    return {"valid_bedroc": sums["bedroc_sum"] / count,
            "valid_auc": sums["auc_sum"] / count,
            "valid_top1_acc": sums["top1_sum"] / count,
            "valid_top3_acc": sums["top3_sum"] / count,
            "valid_loss": sums["loss_sum"] / count,
            "valid_queries": int(count)}
