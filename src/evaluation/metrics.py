"""Evaluation metrics module for citation faithfulness, hallucination profiling,
and Pareto frontier generation.
"""

from __future__ import annotations
import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


def compute_citation_faithfulness(
    generated_claims: List[str], retrieved_contexts: List[str]
) -> float:
    """Estimates the lexical and semantic entailment overlap between
    generated response claims and the retrieved context passages.

    Args:
        generated_claims: Individual factual statements parsed from the generation.
        retrieved_contexts: Target passages injected into the prompt context.

    Returns:
        Scalar faithfulness ratio bounded in [0.0, 1.0].
    """
    if not generated_claims:
        return 1.0
    if not retrieved_contexts:
        return 0.0

    supported_claims = 0
    unified_context = " ".join(retrieved_contexts).lower()

    for claim in generated_claims:
        # Heuristic word containment check (to be replaced by an LLM-as-a-judge in prod)
        claim_words = [w.lower() for w in claim.split() if len(w) > 3]
        if not claim_words:
            continue
        
        overlap = sum(1 for w in claim_words if w in unified_context)
        if (overlap / len(claim_words)) >= 0.60:  # 60% lexical overlap threshold
            supported_claims += 1

    faithfulness = supported_claims / len(generated_claims)
    return round(float(faithfulness), 4)


def evaluate_gating_actions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    risk_scores: np.ndarray,
) -> Dict[str, float]:
    """Computes routing metrics with penalty-weighted risk adjustments.

    Args:
        predictions: Binary decision array (1 = Retrieved, 0 = Skipped).
        ground_truth: Binary ground truth targets.
        risk_scores: Query-level risk magnitudes [0, 1].

    Returns:
        Structured metrics payload containing standard and risk-aware metrics.
    """
    total = len(ground_truth)
    if total == 0:
        return {}

    tp = np.sum((predictions == 1) & (ground_truth == 1))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    tn = np.sum((predictions == 0) & (ground_truth == 0))

    # Risk-Weighted False Negative Rate: FN cases scaled by their query risk magnitude.
    # This is the most critical metric for safety.
    risk_weighted_fn = np.sum(((predictions == 0) & (ground_truth == 1)) * risk_scores)
    total_risk_positives = np.sum((ground_truth == 1) * risk_scores) + 1e-8

    return {
        "accuracy": float((tp + tn) / total),
        "retrieval_rate": float((tp + fp) / total),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "risk_weighted_fn_rate": round(float(risk_weighted_fn / total_risk_positives), 4),
        "unnecessary_retrieval_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }


def compute_pareto_frontier(
    points: List[Dict[str, float]], x_key: str = "latency", y_key: str = "accuracy"
) -> List[Dict[str, float]]:
    """Identifies the Pareto-optimal frontier operating points.
    Optimizes for minimizing x (e.g., latency, cost) while maximizing y (e.g., accuracy).

    Args:
        points: Collection of evaluation runs containing metrics.
        x_key: Metric to minimize along the horizontal axis.
        y_key: Metric to maximize along the vertical axis.

    Returns:
        Filtered list of Pareto-efficient operating coordinates.
    """
    # Sort primarily by ascending cost/latency, secondarily by descending performance
    sorted_pts = sorted(points, key=lambda p: (p[x_key], -p[y_key]))
    pareto_front: List[Dict[str, float]] = []
    cur_max_y = -np.inf

    for pt in sorted_pts:
        # A point is on the Pareto frontier if it offers higher accuracy than any 
        # previously seen point (which, due to sorting, has lower or equal latency).
        if pt[y_key] > cur_max_y:
            pareto_front.append(pt)
            cur_max_y = pt[y_key]

    logger.debug(
        "Pareto Frontier identified: %d optimal points from %d configurations",
        len(pareto_front),
        len(points),
    )
    return pareto_front