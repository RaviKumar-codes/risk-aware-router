"""Gating policy optimization and decision thresholding."""

from __future__ import annotations
import logging
import torch
from typing import Dict, Union

logger = logging.getLogger(__name__)

class AdaptiveDecisionGate:
    """
    Translates router model probabilities and risk scores into a definitive 
    routing action (1: Retrieve, 0: Parametric).
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        dynamic_scaling: bool = True,
        risk_sensitivity: float = 0.3,
        hard_override_threshold: float = 0.85,
    ) -> None:
        """
        Args:
            base_threshold: Baseline probability threshold to trigger retrieval.
            dynamic_scaling: If True, lowers the threshold as query risk increases.
            risk_sensitivity: How much the risk score depresses the threshold (lambda).
            hard_override_threshold: If any individual risk score exceeds this, 
                                     bypass the model and force retrieval.
        """
        self.base_threshold = base_threshold
        self.dynamic_scaling = dynamic_scaling
        self.risk_sensitivity = risk_sensitivity
        self.hard_override_threshold = hard_override_threshold

    def __call__(
        self, 
        router_logits: torch.Tensor, 
        risk_matrix: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, dict]]:
        """
        Applies the decision policy across a batch.

        Args:
            router_logits: Unnormalized logits from the router model, shape (batch, 1).
            risk_matrix: Aggregate risk scores, shape (batch, risk_dim).

        Returns:
            Dict containing the final binary decisions and policy metadata.
        """
        probs = torch.sigmoid(router_logits).squeeze(-1)
        max_risks, _ = torch.max(risk_matrix, dim=-1)

        # 1. Calculate the active threshold per query
        if self.dynamic_scaling:
            # As risk increases, the threshold to retrieve drops.
            # e.g., base 0.5 - (0.3 * 0.9 risk) = 0.23 threshold
            active_thresholds = self.base_threshold - (self.risk_sensitivity * max_risks)
            active_thresholds = torch.clamp(active_thresholds, min=0.05, max=0.95)
        else:
            active_thresholds = torch.full_like(probs, self.base_threshold)

        # 2. Base Model Decision
        model_decisions = (probs >= active_thresholds).int()

        # 3. Hard Safety Override
        # Force retrieval if any single risk dimension exceeds the critical safety threshold
        safety_overrides = (max_risks >= self.hard_override_threshold).int()
        
        # Final decision: Retrieve if the model says so OR the safety override triggered
        final_decisions = torch.bitwise_or(model_decisions, safety_overrides)

        override_count = (safety_overrides > model_decisions).sum().item()
        if override_count > 0:
            logger.debug("Safety override forced retrieval on %d queries.", override_count)

        return {
            "decisions": final_decisions,
            "probabilities": probs,
            "active_thresholds": active_thresholds,
            "overrides_triggered": safety_overrides,
        }