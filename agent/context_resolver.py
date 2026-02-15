"""
Context Resolver - Refines search queries based on previous results.
"""
import logging
import re
from typing import Optional
from models.strategist_models import ExecutionStep
from utils.llm_client import create_llm_client
from prompts.context_resolver_prompts import (
    CONTEXT_RESOLVER_SYSTEM_PROMPT,
    build_user_prompt
)
from config import config

logger = logging.getLogger(__name__)


class ContextResolver:
    """Refines search queries using context from previous results."""
    
    def __init__(self):
        # Get context resolver-specific LLM config
        resolver_config = config.get_context_resolver_llm_config()
        
        # Create LLM client for context resolver
        self.llm_client = create_llm_client(
            provider=resolver_config['provider'],
            api_key=resolver_config['api_key'],
            model=resolver_config['model'],
            temperature=resolver_config['temperature']
        )
        
        logger.info(f"ContextResolver initialized with {resolver_config['provider']}/{resolver_config['model']}")
    
    async def add_context(
        self,
        current_step: ExecutionStep,
        previous_context: str,
        conversation_history: Optional[list[dict]],
        max_retries: int = 3
    ) -> ExecutionStep:
        """
        Refine search queries based on previous results and context.
        
        Args:
            current_step: The step to refine
            previous_context: Concatenated refined data from dependent steps
            conversation_history: Conversation history for context
            max_retries: Maximum number of retry attempts for LLM calls
            
        Returns:
            ExecutionStep with refined search queries
        """
        logger.info(f"Augmenting step {current_step.step_id} with context")
        
        # Convert ExecutionStep to dict for JSON serialization
        step_dict = current_step.model_dump()
        
        # Build user prompt using the prompt builder
        user_prompt = build_user_prompt(
            current_step=step_dict,
            previous_context=previous_context,
            conversation_history=conversation_history
        )
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                messages = [
                    {"role": "system", "content": CONTEXT_RESOLVER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = await self.llm_client.generate(messages)
                print(f"\n\nresponse: {response}")
                # Extract JSON from <response> XML tag
                response_match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
                if response_match:
                    json_text = response_match.group(1).strip()
                else:
                    # Fallback: use entire response if XML tags not found
                    logger.warning(f"No <response> tag found in LLM output for step {current_step.step_id}, using full response")
                    json_text = response.strip()
                
                # Strip markdown code blocks if present
                json_text = re.sub(r'^```(?:json)?\s*\n?', '', json_text)
                json_text = re.sub(r'\n?```\s*$', '', json_text)
                
                # Parse the augmented ExecutionStep using Pydantic
                augmented_step = ExecutionStep.model_validate_json(json_text)
                
                logger.info(f"Successfully augmented step {augmented_step.step_id} with {len(augmented_step.search_queries)} queries")
                return augmented_step
            
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Step {current_step.step_id} augmentation failed (attempt {retry_count}/{max_retries}), retrying: {e}")
                else:
                    logger.warning(f"Step {current_step.step_id} augmentation failed after {max_retries} retries, returning original step: {e}")
                    return current_step  # Return original step on exhaustion
