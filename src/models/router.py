"""Risk-Aware Router Model Architecture."""

from __future__ import annotations
import torch
import torch.nn as nn

class RiskAwareRouter(nn.Module):
    """
    A lightweight gating network that combines dense query embeddings
    with explicit risk signals to make a retrieval decision.
    """

    def __init__(
        self, 
        embedding_dim: int = 768, 
        risk_dim: int = 4, 
        hidden_dim: int = 256, 
        dropout_prob: float = 0.2
    ) -> None:
        """
        Args:
            embedding_dim: Size of the incoming query embedding (e.g., 768 for DeBERTa/BERT).
            risk_dim: Number of explicit risk features extracted (Numerical, STEM, Curriculum, etc.).
            hidden_dim: Size of the intermediate hidden layer.
            dropout_prob: Dropout probability for regularization.
        """
        super().__init__()
        
        # We concatenate the semantic embedding with our explicit risk scores
        input_dim = embedding_dim + risk_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, 1)  # Outputs a single logit (Retrieve/Not)
        )

    def forward(
        self, 
        query_embeddings: torch.Tensor, 
        risk_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the router.

        Args:
            query_embeddings: Tensor of shape (batch_size, embedding_dim)
            risk_vectors: Tensor of shape (batch_size, risk_dim) containing [0, 1] bounded risk scores.

        Returns:
            logits: Unnormalized routing logits of shape (batch_size, 1). 
                    Apply sigmoid during inference to get probabilities.
        """
        # Ensure risk vectors are strictly bounded between 0 and 1
        risk_vectors = torch.clamp(risk_vectors, min=0.0, max=1.0)
        
        # Fuse semantic representations with structural risk features
        fused_features = torch.cat([query_embeddings, risk_vectors], dim=-1)
        
        logits = self.classifier(fused_features)
        return logits