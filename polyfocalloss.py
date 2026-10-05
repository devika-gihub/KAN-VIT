"""Polynomial focal-loss variants used in the experiments."""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class PolyLoss(_Loss):
    def __init__(self, ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = "mean", epsilon: float = 1.0,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.ce_weight = ce_weight
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        classes = input.shape[1]
        one_hot = F.one_hot(target, num_classes=classes).to(input.dtype)

        if self.label_smoothing > 0:
            smoothed = one_hot * (1 - self.label_smoothing) + self.label_smoothing / classes
            ce = F.cross_entropy(
                input, target, weight=self.ce_weight, reduction="none",
                label_smoothing=self.label_smoothing,
            )
            one_minus_pt = torch.sum(smoothed * (1 - F.softmax(input, dim=-1)), dim=-1)
        else:
            pt = torch.sum(one_hot * F.softmax(input, dim=-1), dim=-1)
            ce = F.cross_entropy(input, target, weight=self.ce_weight, reduction="none")
            one_minus_pt = 1 - pt

        loss = ce + self.epsilon * one_minus_pt
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class PolyFocalLoss(_Loss):
    """Polynomial extension of focal loss for multi-class classification."""

    def __init__(self, epsilon: float = 1.0, gamma: float = 2.0,
                 alpha: Optional[List[float]] = None,
                 onehot_encoded: bool = False, reduction: str = "mean") -> None:
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.reduction = reduction
        self.onehot_encoded = onehot_encoded
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        classes = input.shape[1]
        if self.onehot_encoded:
            target_onehot = target.to(dtype=input.dtype)
        else:
            target_onehot = F.one_hot(target, num_classes=classes).to(input.dtype)

        ce = F.cross_entropy(input, target, reduction="none")
        p_t = torch.exp(-ce)
        focal = (1 - p_t).pow(self.gamma) * ce

        if self.alpha is not None:
            if self.alpha.numel() != classes:
                raise ValueError("alpha must contain one value per class")
            alpha = self.alpha.to(input.device)
            alpha = alpha / alpha.sum().clamp_min(torch.finfo(alpha.dtype).eps)
            class_weight = alpha.gather(0, target.reshape(-1)).reshape(target.shape)
            focal = focal * class_weight
            poly_term = self.epsilon * (1 - p_t).pow(self.gamma + 1) * class_weight
        else:
            poly_term = self.epsilon * (1 - p_t).pow(self.gamma + 1)

        loss = focal + poly_term
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
