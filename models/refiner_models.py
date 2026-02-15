"""
Pydantic models for the Refiner component.
"""
from pydantic import BaseModel, Field


class RefinerLLMResponse(BaseModel):
    """LLM response from refiner."""
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    extracted_info: str
    source_indices: list[int] = Field(default_factory=list)


class RefineResult(BaseModel):
    """Result from refining search results."""
    score: float = Field(ge=0.0, le=1.0)
    should_retry: bool
    refined_data: str
    reason: str
    source_indices: list[int] = Field(default_factory=list)  # Indices of sources used (0-based) 
