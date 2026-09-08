"""Custom Loss functions for training risk-aware retrieval routers."""

from __future__ import annotations
import torch
import torch.nn as nn

class RiskAwareCostBenefitLoss(nn.Module):
    """
    Cost-sensitive loss that penalizes unnecessary retrievals (latency/API cost)
    and applies heavy asymmetric penalties to false negatives (skipping retrieval
    on queries with high risk).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 3.5,
        gamma: float = 2.0,
        c_retrieval: float = 0.15,
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            alpha: Base scaling factor for positive errors.
            beta: Risk-multiplier penalizing missed retrievals under high stakes.
            gamma: Focusing parameter (akin to Focal Loss) to down-weight easy cases.
            c_retrieval: Baseline penalty factor representing compute/network latency cost.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c_retrieval = c_retrieval
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        risk_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the risk-aware cost-benefit loss.

        Args:
            logits: Output logits of router before sigmoid, shape (batch_size, 1).
            targets: Binary labels (1: Needs external lookup, 0: Parametric suffices),
                     shape (batch_size, 1).
            risk_scores: Calculated external risk metric [0, 1], shape (batch_size, 1).

        Returns:
            Scalar tensor representing the aggregated batch loss.
        """
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        # POSITIVE CASE (Target = 1, Retrieval Required):
        # A missed retrieval on high-risk input attracts a severe penalty via (1 + beta * risk)
        pos_weight = 1.0 + (self.beta * risk_scores)
        focal_pos = torch.pow(1.0 - probs, self.gamma)
        loss_pos = targets * (self.alpha * focal_pos * pos_weight * -torch.log(probs))

        # NEGATIVE CASE (Target = 0, Parametric Sufficient):
        # An unnecessary retrieval incurs the baseline latency/API cost penalty
        focal_neg = torch.pow(probs, self.gamma)
        cost_weight = 1.0 + self.c_retrieval
        loss_neg = (1.0 - targets) * (focal_neg * cost_weight * -torch.log(1.0 - probs))

        total_loss = loss_pos + loss_neg
        return torch.mean(total_loss)