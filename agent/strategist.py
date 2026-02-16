"""
Research Strategy Planner - decides if single or chained research is needed.
"""

import logging
from typing import Optional
from pydantic import ValidationError

from utils.llm_client import create_llm_client
from prompts.research_strategy_prompts import RESEARCH_STRATEGY_SYSTEM_PROMPT, build_user_prompt
from models.strategist_models import ResearchStrategy, ExecutionStep, SearchQuery
from config import config

logger = logging.getLogger(__name__)


class Strategist:
    """Decides the research strategy for a given query"""
    
    def __init__(self):
        # Get strategist-specific LLM config
        strategist_config = config.get_strategist_llm_config()
        
        self.llm_client = create_llm_client(
            provider=strategist_config['provider'],
            api_key=strategist_config['api_key'],
            model=strategist_config['model'],
            temperature=strategist_config['temperature']
        )
        
        logger.info(f"Strategist initialized with {strategist_config['provider']}/{strategist_config['model']}")
    
    async def plan_research_strategy(
        self,
        query: str,
        conversation_history: Optional[list[dict]] = None
    ) -> ResearchStrategy:
        """Analyze query and decide research strategy."""
        
        logger.info(f"Planning research strategy for: {query}")
        
        # Build user prompt with structured XML format
        user_prompt = build_user_prompt(query, conversation_history)
        
        try:
            # Call LLM with messages format
            messages = [
                {"role": "system", "content": RESEARCH_STRATEGY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self.llm_client.generate(messages=messages)
            #(f"\n\nresponse: {response}")
            
            logger.debug(f"LLM strategy response: {response}")
            
            # Extract JSON from <response> tags if present
            if "<response>" in response and "</response>" in response:
                response = response.split("<response>")[1].split("</response>")[0].strip()
            
            # Strip markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                # Remove opening ```json or ```
                response = response.split("\n", 1)[1] if "\n" in response else response[3:]
            if response.endswith("```"):
                # Remove closing ```
                response = response.rsplit("\n```", 1)[0]
            
            # Parse with Pydantic
            try:
                strategy = ResearchStrategy.model_validate_json(response)
                
                logger.info(f"Execution type: {strategy.execution_type}, Reason: {strategy.reason_summary}")
                logger.info(f"Steps: {len(strategy.steps)}")
                
                return strategy
                
            except (ValidationError, ValueError) as e:
                logger.warning(f"Failed to parse strategy response: {e}. Defaulting to single.")
                return ResearchStrategy(
                    execution_type="single",
                    steps=[
                        ExecutionStep(
                            step_id=1,
                            description="Execute search query",
                            action="search",
                            mode="single",
                            depends_on=[],
                            search_queries=[SearchQuery(query=query, purpose="Answer user query")]
                        )
                    ],
                    reason_summary="Failed to determine strategy, defaulting to single research call",
                    confidence=0.5
                )
        
        except Exception as e:
            logger.error(f"Error planning research strategy: {e}")
            # Default to single strategy on error
            return ResearchStrategy(
                execution_type="single",
                steps=[
                    ExecutionStep(
                        step_id=1,
                        description="Execute search query",
                        action="search",
                        mode="single",
                        depends_on=[],
                        search_queries=[SearchQuery(query=query, purpose="Answer user query")]
                    )
                ],
                reason_summary=f"Error during planning: {str(e)}",
                confidence=0.5
            )
