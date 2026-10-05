"""Loss functions used by the classification pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Class-weighted focal loss for classification."""

    def __init__(self, alpha=None, gamma: float = 2.0,
                 reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 2:
            channels = logits.shape[1]
            logits = logits.movedim(1, -1).reshape(-1, channels)
            target = target.reshape(-1)

        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        logits = logits[valid]
        target = target[valid]
        log_prob = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(
            log_prob, target,
            weight=self.alpha,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        loss = (1.0 - pt).pow(self.gamma) * ce

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
