"""
Pydantic models for LLM-as-Judge evaluation.
"""
from pydantic import BaseModel, Field


class StepScore(BaseModel):
    """Score for a single step."""
    score: float = Field(ge=0.0, le=1.0, description="Score between 0 and 1")
    reasoning: str = Field(description="Brief reasoning for the score")


class LLMJudgeResult(BaseModel):
    """Complete LLM judge evaluation."""
    question_id: str
    question: str
    
    # Strategy evaluation
    strategy_score: float = Field(ge=0.0, le=1.0)
    strategy_reasoning: str
    
    # Search evaluation
    search_score: float = Field(ge=0.0, le=1.0)
    search_reasoning: str
    
    # Refinement evaluation
    refinement_score: float = Field(ge=0.0, le=1.0)
    refinement_reasoning: str
    
    # Context resolution evaluation
    context_score: float = Field(ge=0.0, le=1.0)
    context_reasoning: str
    
    # Per-step search scores (combined)
    search_step_scores: list[StepScore]
    avg_search_score: float = Field(ge=0.0, le=1.0)
    
    # Answer evaluation
    answer_score: float = Field(ge=0.0, le=1.0)
    answer_reasoning: str
    
    # Overall
    overall_score: float = Field(ge=0.0, le=1.0)
