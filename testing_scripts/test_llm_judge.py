"""
Simple test for LLM-as-Judge evaluation.
"""
import asyncio
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.orchestrator import ResearchAgent
from evaluation.llm_judge import LLMJudge, format_judge_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_llm_judge():
    """Test LLM judge on a sample question."""
    
    # Initialize
    agent = ResearchAgent()
    judge = LLMJudge()
    
    # Test question (complex multi-hop query)
    question = "What are the main differences between GPT-4 and Claude 3.5 Sonnet in terms of performance and pricing?"
    question_id = "test_comparison"
    
    logger.info(f"Question: {question}\n")
    
    # Run research
    logger.info("Running research...")
    result = await agent.research(query=question, session_id="llm_judge_test")
    
    print(f"\n{'='*80}")
    print(f"RESEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"Query: {question}")
    print(f"Answer: {result.answer.answer[:200]}...")
    print(f"Citations: {len(result.answer.citations)}")
    
    # Use actual strategy and search data from the agent
    strategy_data = result.strategy_data
    search_steps_data = result.search_steps_data
    
    # Citations
    citations = [
        {'title': c.title, 'domain': c.domain, 'url': c.url}
        for c in result.answer.citations
    ]
    
    # Evaluate with LLM judge
    logger.info("\nEvaluating with LLM judge...")
    judge_result = await judge.evaluate(
        question_id=question_id,
        question=question,
        strategy_data=strategy_data,
        search_steps_data=search_steps_data,
        answer=result.answer.answer,
        citations=citations
    )
    
    # Print results
    print("\n" + "="*80)
    print("LLM JUDGE EVALUATION")
    print("="*80)
    print(format_judge_result(judge_result))
    print()


if __name__ == "__main__":
    asyncio.run(test_llm_judge())
