"""
Context Resolver - Refines search queries based on previous results.
"""
import logging
from typing import Optional
from models.strategist_models import ExecutionStep
from utils.llm_client import create_llm_client
from prompts.context_resolver import CONTEXT_RESOLVER_SYSTEM_PROMPT
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
    
    async def refine_queries(
        self,
        current_step: ExecutionStep,
        previous_results: list[dict],
        conversation_history: Optional[list[dict]],
        original_query: str
    ) -> list[str]:
        """
        Refine search queries based on previous results and context.
        
        Args:
            current_step: The step to generate queries for
            previous_results: Refined data from dependent steps
            conversation_history: Conversation history for context
            original_query: Original user query
            
        Returns:
            List of refined query strings
        """
        logger.info(f"Refining queries for step {current_step.step_id}")
        
        # Build summary from previous refined data
        data_summary = ""
        if previous_results:
            for idx, data in enumerate(previous_results, 1):
                if isinstance(data, dict) and "refined_data" in data:
                    info = data["refined_data"]
                else:
                    info = str(data)
                data_summary += f"\n[{idx}] {info[:300]}..."
        
        # Build conversation context
        conv_context = ""
        if conversation_history:
            for msg in conversation_history[-3:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:200]
                conv_context += f"{role}: {content}\n"
        
        user_prompt = f"""Original Query: {original_query}

Conversation History:
{conv_context if conv_context else "(No previous conversation)"}

Previous Results:
{data_summary if data_summary else "(No previous results)"}

Current Step: {current_step.description}
Generate {len(current_step.search_queries)} specific search queries.
Return ONLY the queries, one per line."""
        
        try:
            messages = [
                {"role": "system", "content": CONTEXT_RESOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self.llm_client.generate(messages)
            queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
            
            # Pad with original queries if needed
            if len(queries) < len(current_step.search_queries):
                for sq in current_step.search_queries[len(queries):]:
                    queries.append(sq.query)
            
            return queries[:len(current_step.search_queries)]
        
        except Exception as e:
            logger.error(f"Error generating refined queries: {e}")
            return [sq.query for sq in current_step.search_queries]
