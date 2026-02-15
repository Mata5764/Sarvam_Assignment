"""
Simple test script for Strategist.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.strategist import Strategist


async def test_strategist():
    """Test Strategist."""
    
    strategist = Strategist()
    
    # Test query - CHANGE THIS TO TEST DIFFERENT SCENARIOS
    query = "Can you please give me an analysis of Cristiano Ronaldo's performance in the season when he won the champions league for Manchester United?"
    history = None
    
    strategy = await strategist.plan_research_strategy(
        query=query,
        conversation_history=history
    )
    
    print(f"\nExecution Type: {strategy.execution_type}")
    print(f"Confidence: {strategy.confidence}")
    print(f"Reason: {strategy.reason_summary}")
    print(f"\nSteps ({len(strategy.steps)}):")
    for step in strategy.steps:
        print(f"  {step.step_id}. {step.description} (action={step.action}, mode={step.mode})")
        if step.search_queries:
            for q in step.search_queries:
                print(f"      - {q.query}")


if __name__ == "__main__":
    asyncio.run(test_strategist())
