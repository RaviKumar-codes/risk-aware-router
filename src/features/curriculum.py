"""Curriculum specificity and assessment stakes extraction module."""

from __future__ import annotations
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CurriculumSpecificityExtractor:
    """
    Evaluates whether a query requires strict alignment with a specific
    educational board, syllabus standard, or high-stakes assessment.
    """

    def __init__(self) -> None:
        # Keywords indicating specific educational frameworks or strict assessments
        self.board_patterns = re.compile(
            r"\b(ncert|cbse|icse|igcse|ap|ib|sat|a-level|o-level|jee|neet)\b", 
            re.IGNORECASE
        )
        self.grade_patterns = re.compile(
            r"\b(grade \d+|class \d+|year \d+|semester \d+)\b", 
            re.IGNORECASE
        )
        self.assessment_keywords = {
            "exam", "syllabus", "past paper", "rubric", "board question", 
            "marks scheme", "standardized test"
        }

    def score(self, query: str) -> Dict[str, Any]:
        """Calculates the curriculum specificity risk score [0.0, 1.0].

        Args:
            query: Input prompt.

        Returns:
            Dict containing the scalar score and detected flags.
        """
        if not query or not query.strip():
            return {"score": 0.0, "board_hits": [], "assessment_hits": []}

        normalized = query.lower()
        
        boards = self.board_patterns.findall(normalized)
        grades = self.grade_patterns.findall(normalized)
        
        words = set(re.findall(r"\b[a-zA-Z_ ]+\b", normalized))
        assessments = [kw for kw in self.assessment_keywords if kw in normalized]

        # Scoring logic: 
        # Mentioning a specific board/framework is a strong signal for retrieval (0.4)
        # Mentioning a grade level adds specificity (0.2)
        # Mentioning assessment/exam contexts adds stakes (0.3)
        
        score = 0.0
        if boards:
            score += 0.4
        if grades:
            score += 0.2
        if assessments:
            score += 0.3
            
        # Cap at 1.0
        final_score = min(score, 1.0)

        logger.debug(
            "Curriculum Check - Score: %.3f | Boards: %s | Assessments: %s", 
            final_score, boards, assessments
        )

        return {
            "score": round(float(final_score), 4),
            "board_hits": boards,
            "grade_hits": grades,
            "assessment_hits": assessments,
        }