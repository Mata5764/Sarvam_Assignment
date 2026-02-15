"""
Pydantic models for the AnswerGenerator component.
"""
from pydantic import BaseModel


class Citation(BaseModel):
    """Represents a source citation."""
    title: str
    domain: str
    url: str


class ResearchAnswer(BaseModel):
    """Complete answer with citations."""
    answer: str
    citations: list[Citation]
