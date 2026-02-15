"""
Answer generator that creates citation-grounded responses from refined data.
"""
import logging
from typing import Optional
from dataclasses import dataclass
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a source citation."""
    title: str
    domain: str
    url: str


@dataclass
class ResearchAnswer:
    """Complete answer with citations."""
    answer: str
    citations: list[Citation]
    confidence: str  # "high", "medium", "low"
    conflicts_detected: bool


class AnswerGenerator:
    """Generates final answers from refined research data."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        logger.info("AnswerGenerator initialized")
    
    async def generate_answer(
        self,
        query: str,
        refined_data: list[dict]
    ) -> ResearchAnswer:
        """
        Generate answer from refined data.
        
        Args:
            query: Original user query
            refined_data: List of refined data dicts from refiner
            
        Returns:
            ResearchAnswer with answer text and citations
        """
        logger.info(f"Generating answer for query: {query}")
        
        if not refined_data:
            return ResearchAnswer(
                answer="I couldn't find relevant information to answer this question.",
                citations=[],
                confidence="low",
                conflicts_detected=False
            )
        
        # Build context from refined data
        context_text = self._build_context_from_refined_data(refined_data)
        
        # Generate answer using LLM
        answer_text = await self._generate_answer_text(query, context_text)
        
        # Extract citations
        citations = self._extract_citations(refined_data)
        
        # Detect conflicts (simplified)
        conflicts = self._detect_conflicts(refined_data)
        
        # Determine confidence
        confidence = self._calculate_confidence(refined_data)
        
        logger.info(f"Generated answer with {len(citations)} citations")
        
        return ResearchAnswer(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            conflicts_detected=conflicts
        )
    
    def _build_context_from_refined_data(self, refined_data: list[dict]) -> str:
        """
        Build context text from refined data.
        
        Each dict in refined_data has structure:
        {
            "refined_data": str,  # Extracted text
            "sources": [...],
            "query": str,
            "score": float
        }
        """
        context_parts = []
        
        for idx, data in enumerate(refined_data, 1):
            # Extract the refined text from the structured dict
            if isinstance(data, dict) and "refined_data" in data:
                info = data["refined_data"]
            else:
                info = str(data)
            
            context_parts.append(f"Source {idx}: {info[:600]}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_answer_text(self, query: str, context_text: str) -> str:
        """Generate answer text using LLM."""
        system_prompt = """You are a helpful research assistant. Generate a comprehensive answer based on the provided sources.
Include inline citations using [Source N] format.
If sources conflict, mention the disagreement.
If information is insufficient, say so."""

        user_prompt = f"""Question: {query}

Research findings:
{context_text}

Generate a well-structured answer with citations."""

        try:
            answer = await self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.4,
                max_tokens=1000
            )
            return answer.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error generating the answer."
    
    def _extract_citations(self, refined_data: list[dict]) -> list[Citation]:
        """Extract citations from refined data."""
        citations = []
        seen_urls = set()
        
        for data in refined_data:
            if "sources" in data:
                for source in data["sources"]:
                    url = source.get("url", "")
                    if url and url not in seen_urls:
                        citations.append(Citation(
                            title=source.get("title", "Unknown"),
                            domain=self._extract_domain(url),
                            url=url
                        ))
                        seen_urls.add(url)
        
        return citations
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url
    
    def _detect_conflicts(self, refined_data: list[dict]) -> bool:
        """Detect if there are conflicts in the data (simplified)."""
        # Simple heuristic: check if "conflict" or "disagree" appears
        for data in refined_data:
            data_str = str(data).lower()
            if "conflict" in data_str or "disagree" in data_str or "however" in data_str:
                return True
        return False
    
    def _calculate_confidence(self, refined_data: list[dict]) -> str:
        """Calculate confidence level based on refined data."""
        if not refined_data:
            return "low"
        
        # Check if we have multiple sources
        total_sources = sum(
            len(data.get("sources", [])) for data in refined_data
        )
        
        if total_sources >= 3:
            return "high"
        elif total_sources >= 2:
            return "medium"
        else:
            return "low"
