"""
Answer generator that creates citation-grounded responses from refined data.
"""
import logging
from typing import Optional
from utils.llm_client import create_llm_client
from models.answer_generator_models import Citation, ResearchAnswer
from prompts.answer_generator_prompts import (
    ANSWER_GENERATOR_SYSTEM_PROMPT,
    build_user_prompt
)
from config import config

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates final answers from refined research data."""
    
    def __init__(self):
        # Get answer generator-specific LLM config
        generator_config = config.get_answer_generator_llm_config()
        
        # Create LLM client for answer generator
        self.llm = create_llm_client(
            provider=generator_config['provider'],
            api_key=generator_config['api_key'],
            model=generator_config['model'],
            temperature=generator_config['temperature']
        )
        
        logger.info(f"AnswerGenerator initialized with {generator_config['provider']}/{generator_config['model']}")
    
    async def generate_answer(
        self,
        query: str,
        refined_data: list[dict],
        max_retries: int = 3
    ) -> ResearchAnswer:
        """
        Generate answer from refined data.
        
        Args:
            query: Original user query
            refined_data: List of refined data dicts from refiner
            max_retries: Maximum number of retry attempts for LLM calls
            
        Returns:
            ResearchAnswer with answer text and citations
        """
        logger.info(f"Generating answer for query: {query}")
        
        if not refined_data:
            return ResearchAnswer(
                answer="I couldn't find relevant information to answer this question.",
                citations=[]
            )
        
        # Build context from refined data
        context_text = self._build_context_from_refined_data(refined_data)
        
        # Generate answer using LLM with retries
        answer_text = await self._generate_answer_text(query, context_text, max_retries)
        
        # Extract citations
        citations = self._extract_citations(refined_data)
        
        logger.info(f"Generated answer with {len(citations)} citations")
        
        return ResearchAnswer(
            answer=answer_text,
            citations=citations
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
    
    async def _generate_answer_text(self, query: str, context_text: str, max_retries: int = 3) -> str:
        """Generate answer text using LLM with retry logic."""
        
        # Build user prompt using the prompt builder
        user_prompt = build_user_prompt(query, context_text)

        retry_count = 0
        while retry_count < max_retries:
            try:
                messages = [
                    {"role": "system", "content": ANSWER_GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                
                answer = await self.llm.generate(messages)
                return answer.strip()
            
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Answer generation failed (attempt {retry_count}/{max_retries}), retrying: {e}")
                else:
                    logger.warning(f"Answer generation failed after {max_retries} retries: {e}")
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
