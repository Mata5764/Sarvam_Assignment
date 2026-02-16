"""
LLM-as-Judge Evaluation for Deep Research Agent.

Single comprehensive evaluation of the complete research workflow.
"""
import re
import logging
from typing import Optional

from utils.llm_client import create_llm_client
from config import config
from models.llm_judge_models import StepScore, LLMJudgeResult
from prompts import LLM_JUDGE_SYSTEM_PROMPT, build_evaluation_prompt

logger = logging.getLogger(__name__)


class LLMJudge:
    """LLM-based evaluator for research quality."""
    
    def __init__(self):
        """Initialize LLM judge with dedicated client."""
        llm_config = config.get_llm_judge_config()
        self.llm_client = create_llm_client(
            provider=llm_config['provider'],
            api_key=llm_config['api_key'],
            model=llm_config['model'],
            temperature=llm_config['temperature']
        )
        logger.info(f"LLMJudge initialized with {llm_config['provider']}/{llm_config['model']}")
    
    async def evaluate(
        self,
        question_id: str,
        question: str,
        strategy_data: dict,
        search_steps_data: list[dict],
        answer: str,
        citations: list[dict]
    ) -> LLMJudgeResult:
        """
        Evaluate complete research workflow with a single LLM call.
        
        Args:
            question_id: Question identifier
            question: The research question
            strategy_data: {type, steps, reasoning}
            search_steps_data: [{description, queries, refined_data}, ...]
            answer: Final answer
            citations: List of citation dicts
        
        Returns:
            LLMJudgeResult with all scores and reasoning
        """
        # Build comprehensive prompt using the new structured prompt
        prompt = build_evaluation_prompt(
            question=question,
            strategy_data=strategy_data,
            search_steps_data=search_steps_data,
            answer=answer,
            citations=citations
        )
        
        # Single LLM call with system prompt
        messages = [
            {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        response = await self.llm_client.generate(messages)
        print(response)
        
        # Parse all scores from XML-tagged response
        return self._parse_evaluation_response(
            response=response,
            question_id=question_id,
            question=question,
            search_steps_data=search_steps_data
        )
    
    def _parse_evaluation_response(
        self,
        response: str,
        question_id: str,
        question: str,
        search_steps_data: list[dict]
    ) -> LLMJudgeResult:
        """Parse XML-tagged evaluation response with final JSON scores."""
        import json
        
        def extract_tag(text: str, tag: str) -> str:
            """Extract content from XML tag."""
            match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
            return match.group(1).strip() if match else ""
        
        def clean_reasoning(text: str) -> str:
            """Clean reasoning text."""
            # Remove instruction text
            text = re.sub(r'^Evaluate.*?:', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'^-.*?\n', '', text, flags=re.MULTILINE).strip()
            # Collapse whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove bracketed placeholders
            text = re.sub(r'\[.*?\]', '', text).strip()
            
            if not text or text.lower().startswith("n/a"):
                return "No detailed reasoning provided"
            elif len(text) > 250:
                return text[:250] + "..."
            return text
        
        # Extract reasoning from each section
        strategy_reasoning = clean_reasoning(extract_tag(response, 'strategy_evaluation'))
        search_reasoning = clean_reasoning(extract_tag(response, 'search_evaluation'))
        refinement_reasoning = clean_reasoning(extract_tag(response, 'refinement_evaluation'))
        context_reasoning = clean_reasoning(extract_tag(response, 'context_resolution_evaluation'))
        answer_reasoning = clean_reasoning(extract_tag(response, 'answer_evaluation'))
        
        # Extract JSON scores from <response> tag
        response_tag = extract_tag(response, 'response')
        
        if response_tag:
            try:
                scores = json.loads(response_tag)
                strategy_score = scores.get('strategy_score', 0.0)
                search_score = scores.get('search_score', 0.0)
                refinement_score = scores.get('refinement_score', 0.0)
                context_score = scores.get('context_score', 0.0)
                answer_score = scores.get('answer_score', 0.0)
                overall_score = scores.get('overall_score', 0.0)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON scores from <response> tag, using defaults")
                strategy_score = search_score = refinement_score = context_score = answer_score = 0.5
                overall_score = 0.5
        else:
            logger.warning("No <response> tag found, using defaults")
            strategy_score = search_score = refinement_score = context_score = answer_score = 0.5
            overall_score = 0.5
        
        # Calculate weighted average if overall_score is 0
        if overall_score == 0.0:
            overall_score = (
                strategy_score * 0.20 +
                search_score * 0.20 +
                refinement_score * 0.20 +
                context_score * 0.10 +
                answer_score * 0.30
            )
        
        # Create per-step scores combining all components
        search_step_scores = []
        for i, step_data in enumerate(search_steps_data, 1):
            # Combine search, refinement, and context scores for this step
            step_score = (search_score + refinement_score) / 2
            if 'query_refinement' in step_data and context_score > 0:
                step_score = (step_score * 2 + context_score) / 3
            
            components = [
                f"Search: {search_score:.2f}",
                f"Refine: {refinement_score:.2f}"
            ]
            if 'query_refinement' in step_data:
                components.append(f"Context: {context_score:.2f}")
            
            step_reasoning = " | ".join(components)
            
            search_step_scores.append(StepScore(
                score=step_score,
                reasoning=step_reasoning
            ))
        
        avg_search_score = sum(s.score for s in search_step_scores) / len(search_step_scores) if search_step_scores else 0.0
        
        return LLMJudgeResult(
            question_id=question_id,
            question=question,
            strategy_score=strategy_score,
            strategy_reasoning=strategy_reasoning,
            search_score=search_score,
            search_reasoning=search_reasoning,
            refinement_score=refinement_score,
            refinement_reasoning=refinement_reasoning,
            context_score=context_score,
            context_reasoning=context_reasoning,
            search_step_scores=search_step_scores,
            avg_search_score=avg_search_score,
            answer_score=answer_score,
            answer_reasoning=answer_reasoning,
            overall_score=overall_score
        )


def format_judge_result(result: LLMJudgeResult) -> str:
    """Format LLM judge result for display."""
    output = []
    output.append(f"Question: {result.question}")
    output.append(f"\n{'='*70}")
    output.append("COMPONENT EVALUATIONS")
    output.append(f"{'='*70}")
    
    output.append(f"\n1. Strategy Planning: {result.strategy_score:.2f}/1.0")
    output.append(f"   {result.strategy_reasoning}")
    
    output.append(f"\n2. Web Search Quality: {result.search_score:.2f}/1.0")
    output.append(f"   {result.search_reasoning}")
    
    output.append(f"\n3. Content Refinement: {result.refinement_score:.2f}/1.0")
    output.append(f"   {result.refinement_reasoning}")
    
    output.append(f"\n4. Context Resolution: {result.context_score:.2f}/1.0")
    output.append(f"   {result.context_reasoning}")
    
    output.append(f"\n5. Answer Generation: {result.answer_score:.2f}/1.0")
    output.append(f"   {result.answer_reasoning}")
    
    output.append(f"\n{'='*70}")
    output.append(f"OVERALL SCORE: {result.overall_score:.2f}/1.0")
    output.append(f"{'='*70}")
    output.append("Weights: Strategy 20% | Search 20% | Refinement 20% | Context 10% | Answer 30%")
    
    return "\n".join(output)
