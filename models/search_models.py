"""
Pydantic models for Search component.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a single search result."""
    title: str
    url: str
    snippet: str
    content: Optional[str] = ""  # Full page content (if available from provider)
    relevance_score: Optional[float] = None
    published_date: Optional[str] = None
    domain: Optional[str] = None
