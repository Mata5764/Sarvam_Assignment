"""
Refiner - Validates search results and extracts relevant information.
Acts as a quality gate after each search, can trigger retries if needed.
"""
import logging
from typing import Optional
from agent.search import SearchResult
from utils.llm_client import create_llm_client
from models.refiner_models import RefineResult, RefinerLLMResponse
from prompts.refiner_prompts import REFINER_SYSTEM_PROMPT
from pydantic import ValidationError
from config import config
import json
import re

logger = logging.getLogger(__name__)


class Refiner:
    """Validates and refines search results."""
    
    def __init__(self, max_retries: int = 1):
        # Get refiner-specific LLM config
        refiner_config = config.get_refiner_llm_config()
        
        # Create LLM client for refiner
        self.llm_client = create_llm_client(
            provider=refiner_config['provider'],
            api_key=refiner_config['api_key'],
            model=refiner_config['model'],
            temperature=refiner_config['temperature']
        )
        
        self.max_retries = max_retries
        logger.info(f"Refiner initialized with {refiner_config['provider']}/{refiner_config['model']}")
    
    async def refine_search_results(
        self,
        query: str,
        results: list[SearchResult],
        retry_count: int = 3
    ) -> RefineResult:
        """
        Refine search results - validate, score, extract relevant info.
        
        Args:
            query: The search query that was executed
            results: Search results (pre-filtered to have substantial content)
            retry_count: How many times we've retried this query
            
        Returns:
            RefineResult with score, retry decision, and extracted data
        """
        logger.info(f"Refining {len(results)} results for query: '{query}' (attempt {retry_count + 1})")
        
        # Use LLM to analyze and extract relevant info
        try:
            refined = await self._llm_refine(query, results)

            print(f"\n\nrefined: {refined}")
            # Decide if we should retry
            should_retry = (
                refined["score"] < 0.5 and 
                retry_count < self.max_retries
            )
            
            return RefineResult(
                score=refined["score"],
                should_retry=should_retry,
                refined_data=refined["extracted_info"],
                reason=refined["reason"],
                source_indices=refined.get("source_indices", list(range(len(results))))  # Default to all if not specified
            )
        
        except Exception as e:
            # Decide if we should retry on API error
            should_retry = retry_count < self.max_retries
            
            # Log appropriately based on whether we'll retry
            if should_retry:
                logger.info(f"API error for query '{query}' (will retry): {e}")
            else:
                logger.warning(f"API error for query '{query}' (max retries reached): {e}")
            
            # On error, use basic extraction but retry if within limits
            return RefineResult(
                score=0.5,  # Neutral score since we don't have LLM judgment
                should_retry=should_retry,
                refined_data=self._extract_basic_info(query, results),
                reason=f"API error (attempt {retry_count + 1}/{self.max_retries + 1}): {str(e)}",
                source_indices=list(range(min(3, len(results))))  # Use first 3 sources as fallback
            )
    
    async def _llm_refine(self, query: str, results: list[SearchResult]) -> dict:
        """Use LLM to validate and extract relevant information."""
        
        # Build search results summary
        results_summary = ""
        for idx, result in enumerate(results[:5], 1):  # Top 5 results
            content_preview = result.content[:500] if result.content else result.snippet
            results_summary += f"\n[{idx}] {result.title}\n{content_preview}...\n"
        
        user_prompt = f"""Query: {query}

Search Results:
{results_summary}

Analyze these results and extract relevant information."""

        try:
            messages = [
                {"role": "system", "content": REFINER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self.llm_client.generate(messages)
            
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            # Parse and validate JSON response using Pydantic
            llm_response = RefinerLLMResponse.model_validate_json(response)
            
            # Convert to dict for compatibility
            refined = llm_response.model_dump()
            
            return refined
        
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}, using fallback")
            return {
                "score": 0.6,
                "reason": "LLM parsing failed, using basic extraction",
                "extracted_info": self._extract_basic_info(query, results),
                "source_indices": list(range(min(3, len(results))))  # Use first 3 sources
            }
    
    def _extract_basic_info(self, query: str, results: list[SearchResult]) -> str:
        """Fallback: Basic extraction without LLM."""
        # Concatenate top snippets/content
        info_parts = []
        
        for idx, result in enumerate(results[:3], 1):
            content = result.content[:400] if result.content else result.snippet
            if content:
                info_parts.append(f"[{idx}] {result.title}: {content}")
        
        return "\n\n".join(info_parts) if info_parts else "No substantial information found."
