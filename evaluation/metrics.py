"""
Evaluation metrics for assessing research agent performance.

Key metrics:
1. Citation Quality: Are sources cited properly?
2. Grounding: Is the answer grounded in retrieved sources?
3. Completeness: Does the answer address all parts of the question?
4. Conflict Handling: Are conflicts properly identified?
5. Uncertainty Expression: Is uncertainty expressed when appropriate?
"""
import re
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluating a single response."""
    question_id: str
    question: str
    answer: str
    
    # Citation metrics
    num_citations: int
    has_citations: bool
    citation_format_correct: bool
    
    # Content metrics
    contains_expected_elements: bool
    expected_elements_found: list[str]
    expected_elements_missing: list[str]
    
    # Conflict handling
    should_note_conflict: bool
    conflict_noted: bool
    
    # Uncertainty handling
    should_express_uncertainty: bool
    uncertainty_expressed: bool
    
    # Confidence
    confidence_level: str
    confidence_appropriate: bool
    
    # Overall scores (0-1)
    citation_quality_score: float
    grounding_score: float
    completeness_score: float
    overall_score: float
    
    # Additional info
    notes: str = ""


class EvaluationMetrics:
    """Evaluator for research agent responses."""
    
    def __init__(self):
        self.citation_pattern = r'\[([^\]]+?)\s*—\s*([^\]]+?)\]\(([^)]+)\)'
        self.uncertainty_phrases = [
            'unclear', 'uncertain', 'limited information',
            'not enough', 'insufficient', 'may', 'might',
            'possibly', 'appears to', 'seems to', 'likely',
            'no definitive', 'conflicting', 'varies'
        ]
        self.conflict_phrases = [
            'conflict', 'disagree', 'however', 'while source',
            'contradicts', 'differing', 'different sources',
            'on the other hand', 'alternatively', 'disputed'
        ]
    
    def evaluate(
        self,
        question_id: str,
        question: str,
        answer: str,
        citations: list[dict],
        confidence: str,
        expected_elements: list[str],
        should_note_conflict: bool = False,
        should_express_uncertainty: bool = False
    ) -> EvaluationResult:
        """
        Evaluate a single response.
        
        Args:
            question_id: Question identifier
            question: The question asked
            answer: Generated answer
            citations: List of citation dicts
            confidence: Confidence level (high/medium/low)
            expected_elements: Expected elements in the answer
            should_note_conflict: Whether conflicts should be noted
            should_express_uncertainty: Whether uncertainty should be expressed
            
        Returns:
            EvaluationResult with detailed metrics
        """
        # Citation metrics
        num_citations = len(citations)
        has_citations = num_citations > 0
        citation_format_correct = self._check_citation_format(answer, citations)
        
        # Content metrics
        expected_found, expected_missing = self._check_expected_elements(
            answer, expected_elements
        )
        contains_expected_elements = len(expected_found) > 0
        
        # Conflict handling
        conflict_noted = self._check_conflict_noted(answer)
        
        # Uncertainty handling
        uncertainty_expressed = self._check_uncertainty_expressed(answer)
        
        # Confidence appropriateness
        confidence_appropriate = self._check_confidence_appropriate(
            confidence, num_citations, uncertainty_expressed, should_express_uncertainty
        )
        
        # Calculate scores
        citation_quality_score = self._calculate_citation_quality(
            num_citations, citation_format_correct
        )
        
        grounding_score = self._calculate_grounding_score(
            has_citations, citation_format_correct
        )
        
        completeness_score = self._calculate_completeness_score(
            expected_found, expected_elements
        )
        
        # Conflict/uncertainty bonus
        conflict_bonus = 0.0
        if should_note_conflict and conflict_noted:
            conflict_bonus = 0.1
        if should_express_uncertainty and uncertainty_expressed:
            conflict_bonus = 0.1
        
        # Overall score
        overall_score = (
            citation_quality_score * 0.4 +
            grounding_score * 0.3 +
            completeness_score * 0.3 +
            conflict_bonus
        )
        overall_score = min(overall_score, 1.0)
        
        # Generate notes
        notes = self._generate_notes(
            num_citations, citation_format_correct,
            expected_found, expected_missing,
            conflict_noted, should_note_conflict,
            uncertainty_expressed, should_express_uncertainty,
            confidence_appropriate
        )
        
        return EvaluationResult(
            question_id=question_id,
            question=question,
            answer=answer,
            num_citations=num_citations,
            has_citations=has_citations,
            citation_format_correct=citation_format_correct,
            contains_expected_elements=contains_expected_elements,
            expected_elements_found=expected_found,
            expected_elements_missing=expected_missing,
            should_note_conflict=should_note_conflict,
            conflict_noted=conflict_noted,
            should_express_uncertainty=should_express_uncertainty,
            uncertainty_expressed=uncertainty_expressed,
            confidence_level=confidence,
            confidence_appropriate=confidence_appropriate,
            citation_quality_score=citation_quality_score,
            grounding_score=grounding_score,
            completeness_score=completeness_score,
            overall_score=overall_score,
            notes=notes
        )
    
    def _check_citation_format(self, answer: str, citations: list[dict]) -> bool:
        """Check if citations are properly formatted."""
        if not citations:
            return False
        
        # Find all citations in the answer
        matches = re.findall(self.citation_pattern, answer)
        
        # Should have at least some citations in proper format
        return len(matches) > 0
    
    def _check_expected_elements(
        self, answer: str, expected_elements: list[str]
    ) -> tuple[list[str], list[str]]:
        """Check which expected elements are present in the answer."""
        answer_lower = answer.lower()
        
        found = []
        missing = []
        
        for element in expected_elements:
            if element.lower() in answer_lower:
                found.append(element)
            else:
                missing.append(element)
        
        return found, missing
    
    def _check_conflict_noted(self, answer: str) -> bool:
        """Check if conflicts are noted in the answer."""
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in self.conflict_phrases)
    
    def _check_uncertainty_expressed(self, answer: str) -> bool:
        """Check if uncertainty is expressed in the answer."""
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in self.uncertainty_phrases)
    
    def _check_confidence_appropriate(
        self,
        confidence: str,
        num_citations: int,
        uncertainty_expressed: bool,
        should_express_uncertainty: bool
    ) -> bool:
        """Check if confidence level is appropriate."""
        if should_express_uncertainty:
            # Should be low confidence with uncertainty expression
            return confidence == "low" and uncertainty_expressed
        
        if num_citations >= 2 and not uncertainty_expressed:
            # Should be high confidence
            return confidence == "high"
        
        if num_citations == 0:
            # Should be low confidence
            return confidence == "low"
        
        # Medium is generally appropriate for in-between cases
        return confidence in ["medium", "low"]
    
    def _calculate_citation_quality(
        self, num_citations: int, format_correct: bool
    ) -> float:
        """Calculate citation quality score."""
        if num_citations == 0:
            return 0.0
        
        score = 0.0
        
        # Number of citations (up to 0.6)
        if num_citations >= 3:
            score += 0.6
        elif num_citations == 2:
            score += 0.4
        else:
            score += 0.2
        
        # Format correctness (0.4)
        if format_correct:
            score += 0.4
        
        return score
    
    def _calculate_grounding_score(
        self, has_citations: bool, format_correct: bool
    ) -> float:
        """Calculate grounding score."""
        if not has_citations:
            return 0.0
        
        if format_correct:
            return 1.0
        
        return 0.5
    
    def _calculate_completeness_score(
        self, found: list[str], expected: list[str]
    ) -> float:
        """Calculate completeness score."""
        if not expected:
            return 1.0
        
        return len(found) / len(expected)
    
    def _generate_notes(
        self,
        num_citations: int,
        citation_format_correct: bool,
        expected_found: list[str],
        expected_missing: list[str],
        conflict_noted: bool,
        should_note_conflict: bool,
        uncertainty_expressed: bool,
        should_express_uncertainty: bool,
        confidence_appropriate: bool
    ) -> str:
        """Generate evaluation notes."""
        notes = []
        
        if num_citations == 0:
            notes.append("⚠️ No citations provided")
        elif not citation_format_correct:
            notes.append("⚠️ Citation format needs improvement")
        else:
            notes.append(f"✓ {num_citations} properly formatted citations")
        
        if expected_missing:
            notes.append(f"⚠️ Missing expected elements: {', '.join(expected_missing)}")
        
        if should_note_conflict and not conflict_noted:
            notes.append("⚠️ Should have noted conflicting sources")
        elif conflict_noted:
            notes.append("✓ Properly noted conflicting information")
        
        if should_express_uncertainty and not uncertainty_expressed:
            notes.append("⚠️ Should have expressed uncertainty")
        elif should_express_uncertainty and uncertainty_expressed:
            notes.append("✓ Appropriately expressed uncertainty")
        
        if not confidence_appropriate:
            notes.append("⚠️ Confidence level may not be appropriate")
        
        return " | ".join(notes)


def aggregate_results(results: list[EvaluationResult]) -> dict:
    """
    Aggregate evaluation results across multiple questions.
    
    Returns:
        Dictionary with aggregate metrics
    """
    if not results:
        return {}
    
    total = len(results)
    
    return {
        "total_questions": total,
        "avg_citations": sum(r.num_citations for r in results) / total,
        "citation_rate": sum(1 for r in results if r.has_citations) / total,
        "citation_format_rate": sum(1 for r in results if r.citation_format_correct) / total,
        "completeness_rate": sum(1 for r in results if len(r.expected_elements_found) > 0) / total,
        "conflict_handling_rate": sum(
            1 for r in results 
            if not r.should_note_conflict or r.conflict_noted
        ) / total,
        "uncertainty_handling_rate": sum(
            1 for r in results 
            if not r.should_express_uncertainty or r.uncertainty_expressed
        ) / total,
        "confidence_appropriate_rate": sum(
            1 for r in results if r.confidence_appropriate
        ) / total,
        "avg_citation_quality": sum(r.citation_quality_score for r in results) / total,
        "avg_grounding_score": sum(r.grounding_score for r in results) / total,
        "avg_completeness_score": sum(r.completeness_score for r in results) / total,
        "avg_overall_score": sum(r.overall_score for r in results) / total,
    }
