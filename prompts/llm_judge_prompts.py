"""
Prompts for LLM-as-Judge evaluation.
"""

LLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for research agent workflows.

Your task is to evaluate the complete research process across 5 key components:
1. Strategy Planning - Was the execution plan appropriate?
2. Web Search - Were search queries effective and relevant?
3. Content Refinement - Did the refiner extract quality information?
4. Context Resolution - Were queries enhanced with previous context (if applicable)?
5. Answer Generation - Is the final answer accurate, complete, and well-cited?

For each component, provide:
- A score (0.0 to 1.0)
- Clear reasoning for the score

Be strict but fair. A score of 1.0 should be exceptional."""


def build_evaluation_prompt(
    question: str,
    strategy_data: dict,
    search_steps_data: list[dict],
    answer: str,
    citations: list[dict]
) -> str:
    """Build structured evaluation prompt with COT tags."""
    
    # Format strategy
    strategy_text = f"Type: {strategy_data['type']}\n"
    strategy_text += f"Reasoning: {strategy_data.get('reasoning', 'N/A')}\n"
    strategy_text += f"Confidence: {strategy_data.get('confidence', 'N/A')}\n"
    strategy_text += "Steps:\n"
    for step in strategy_data['steps']:
        queries = ', '.join(q.get('query', '') for q in step.get('search_queries', []))
        strategy_text += f"  - {step.get('description', 'N/A')} (action={step.get('action')}, mode={step.get('mode')})\n"
        if queries:
            strategy_text += f"    Queries: {queries}\n"
    
    # Format search execution with refinement details
    search_text = ""
    has_context_resolution = False
    for i, step_data in enumerate(search_steps_data, 1):
        search_text += f"\nStep {i}: {step_data['description']}\n"
        
        # Check if context resolution was used
        if 'query_refinement' in step_data:
            has_context_resolution = True
            search_text += "  Context Resolution Applied:\n"
            search_text += f"    Original: {step_data['query_refinement']['original']}\n"
            search_text += f"    Refined:  {step_data['query_refinement']['refined']}\n"
        
        # Show refined data
        for query_data in step_data.get('refined_data', []):
            search_text += f"  Query: {query_data.get('query', 'N/A')}\n"
            search_text += f"  Refiner Score: {query_data.get('score', 0):.2f}\n"
            search_text += f"  Refiner Reason: {query_data.get('reason', 'N/A')}\n"
            extracted = query_data.get('refined_data', '')
            if len(extracted) > 200:
                extracted = extracted[:200] + "..."
            search_text += f"  Extracted Info: {extracted}\n"
            
            # Show sources used
            sources = query_data.get('sources', [])
            if sources:
                search_text += f"  Sources Used: {len(sources)} source(s)\n"
    
    # Format citations
    citation_text = f"Total Citations: {len(citations)}\n"
    unique_domains = set(c.get('domain', '') for c in citations)
    citation_text += f"Unique Domains: {len(unique_domains)}\n"
    citation_text += "Citations:\n"
    for i, cit in enumerate(citations[:5], 1):  # Show first 5
        citation_text += f"  {i}. {cit.get('title', 'N/A')} ({cit.get('domain', 'N/A')})\n"
    
    prompt = f"""You are evaluating a research agent's workflow.

<question>{question}</question>

<strategy_plan>
{strategy_text}
</strategy_plan>

<search_execution>
{search_text}
</search_execution>

<answer>
{answer}
</answer>

<citations>
{citation_text}
</citations>

<context_resolution_used>{has_context_resolution}</context_resolution_used>

Evaluate each component with chain-of-thought reasoning.

<strategy_evaluation>
[Evaluate the strategy planning in 2-3 sentences:
 - Was the execution type (single/chain) appropriate for the question?
 - Were the steps logical, complete, and well-structured?
 - Were search queries well-formulated and relevant?]
</strategy_evaluation>

<search_evaluation>
[Evaluate web search quality in 2-3 sentences:
 - Were search queries relevant and specific to the information needs?
 - Did searches return useful, on-topic information?
 - Consider quality across all search steps]
</search_evaluation>

<refinement_evaluation>
[Evaluate content refinement in 2-3 sentences:
 - Did the refiner successfully extract relevant information?
 - Were refiner scores (0.0-1.0) accurate indicators of quality?
 - Was extracted information concise, relevant, and on-topic?]
</refinement_evaluation>

<context_resolution_evaluation>
[Evaluate context resolution in 2-3 sentences (or write "N/A - not used" if skipped):
 - If used: Were queries meaningfully improved with previous context?
 - If used: Did refined queries add specificity and use concrete information?
 - If not used: Was it appropriate to skip (e.g., no dependencies)?]
</context_resolution_evaluation>

<answer_evaluation>
[Evaluate answer generation in 2-3 sentences:
 - Is the answer accurate, complete, and directly addresses the question?
 - Are citations properly used, relevant, and from quality sources?
 - Is the answer well-structured and easy to understand?]
</answer_evaluation>

<overall_assessment>
[Summarize overall research quality in 1-2 sentences]
</overall_assessment>

After completing your evaluation, provide all scores in this JSON format (0.0-1.0 scale):

<response>
{{
  "strategy_score": 0.0,
  "search_score": 0.0,
  "refinement_score": 0.0,
  "context_score": 0.0,
  "answer_score": 0.0,
  "overall_score": 0.0
}}
</response>

IMPORTANT: 
- Write natural reasoning in each XML section (do NOT include the bracketed instructions in your response)
- Provide ONE JSON inside <response> tags at the end with all scores
- Overall score should be weighted: Strategy 20%, Search 20%, Refinement 20%, Context 10%, Answer 30%
- Be strict but fair (1.0 should be exceptional)
"""
    
    return prompt
