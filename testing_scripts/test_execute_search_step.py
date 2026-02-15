"""
Simple test script for _execute_search_step method.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.orchestrator import ResearchAgent
from models.strategist_models import ExecutionStep, SearchQuery


async def test_execute_search_step():
    """Test _execute_search_step."""
    
    agent = ResearchAgent()

    def emit(stage: str, message: str):
        print(f"[{stage}] {message}")
    
    # Create execution step - CHANGE THIS TO TEST DIFFERENT SCENARIOS
    step = ExecutionStep(
        step_id="1",
        description="Find the year Royal Challengers Bangalore won the IPL",
        action="search",
        mode="single",
        search_queries=[
            SearchQuery(query="Royal Challengers Bangalore IPL win year", purpose="Identify the year of the win")
        ],
        depends_on=[]
    )
    
    results = await agent._execute_search_step(
        step=step,
        original_query="Test query",
        conversation_history=None,
        previous_results={},
        emit=emit
    )
    
    print(f"\nReturned {len(results)} result(s):")
    for idx, result in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(f"  Query: {result.get('query')}")
        print(f"  Score: {result.get('score')}")
        print(f"  Sources: {len(result.get('sources', []))}")
        if 'error' in result:
            print(f"  Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(test_execute_search_step())
