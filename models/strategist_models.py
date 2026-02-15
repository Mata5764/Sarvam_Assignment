"""
Pydantic models for the Strategist component.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Individual search query with purpose."""
    query: str
    purpose: str


class ExecutionStep(BaseModel):
    """Single step in the research execution plan."""
    step_id: int
    description: str
    action: Literal["search", "generation"]  # Only these two values allowed
    mode: Literal["single", "parallel"]  # Only these two values allowed
    depends_on: list[int] = Field(default_factory=list)
    search_queries: list[SearchQuery]


class ResearchStrategy(BaseModel):
    """Complete research strategy with execution plan."""
    execution_type: Literal["single", "chain"]  # Only these two values allowed
    steps: list[ExecutionStep] = Field(min_length=1)  # Must have at least 1 step
    reason_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
