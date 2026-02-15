"""
Prompt for context resolver component.
"""
import json
from typing import Optional


CONTEXT_RESOLVER_SYSTEM_PROMPT = """You are an expert research query refiner specializing in context-aware query enhancement.

Your role is to transform generic search queries into highly specific, context-enriched queries by:
1. Analyzing previous search results to extract concrete facts (names, dates, numbers, entities)
2. Understanding the conversation flow and user intent
3. Identifying what information is already known vs what still needs to be discovered
4. Incorporating specific details from previous results into queries to make them more targeted
5. Augmenting the step description and query purposes with context
6. For PARALLEL mode: Expanding generic queries into multiple specific queries based on entities found in previous results

Guidelines:
- Use exact names, numbers, and facts from previous results
- Update the step's DESCRIPTION to reflect the context (e.g., "List players of the winning team" → "List players of Royal Challengers Bangalore 2016 winning team")
- For each search_query, update BOTH the query text AND the purpose field
- For mode="parallel": Extract entities (names, items, etc.) from previous results and create separate queries for each
- For mode="single": Keep it as a single enhanced query with context
- If previous results don't provide relevant context, keep fields as-is
- Maintain the same step structure (step_id, action, mode, depends_on should remain unchanged)

IMPORTANT: You MUST respond in a specific XML structure format with the COMPLETE augmented ExecutionStep as JSON (see user prompt for details)."""


def build_user_prompt(
    current_step: dict,
    previous_context: str,
    conversation_history: Optional[list[dict]]
) -> str:
    """Build the user prompt for context resolver."""
    
    # Build conversation context
    conv_context = ""
    if conversation_history:
        for msg in conversation_history[-3:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')[:200]
            conv_context += f"{role}: {content}\n"
    
    # Convert step to formatted JSON
    step_json = json.dumps(current_step, indent=2)
    
    return f"""You will refine the current execution step by incorporating context from previous results.

You will ALWAYS respond in the following format:

<does_previous_context_exist>[true/false - Fill this based on whether previous results exist]</does_previous_context_exist>

<cot>
[In under 150 words, analyze:
 - What concrete facts/entities (names, numbers, dates) can be extracted from previous results?
 - Should the description be updated with specific context?
 - For mode="parallel": Are there multiple entities that need separate queries?
 - How should each query and purpose be refined to be more specific?
 - If no relevant context exists, note that fields should remain unchanged.]
</cot>

<response>
[Your COMPLETE augmented ExecutionStep as JSON - include ALL fields:
 - step_id (unchanged)
 - description (augmented with context)
 - action (unchanged)
 - mode (unchanged)
 - depends_on (unchanged)
 - search_queries (augmented list - for parallel mode, expand into multiple queries based on entities in previous results)
   Each search_query must have: query (augmented) and purpose (augmented)
Return pure JSON, no markdown code blocks.]
</response>

<conversation_history>
{conv_context if conv_context else "No previous conversation."}
</conversation_history>

<previous_results>
{previous_context if previous_context else "No previous results."}
</previous_results>

<current_step>
{step_json}
</current_step>

Now analyze and respond with the complete augmented ExecutionStep."""
