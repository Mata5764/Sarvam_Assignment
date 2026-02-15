"""
Test script for ContextResolver component.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.context_resolver import ContextResolver
from models.strategist_models import ExecutionStep, SearchQuery


async def test_context_resolver():
    """Test ContextResolver with a sample step."""
    
    # Initialize ContextResolver
    resolver = ContextResolver()
    
    # Sample current step (from strategist - step 3 with parallel mode)
    current_step = ExecutionStep(
        step_id=3,
        description="Fetch career statistics of each player",
        action="search",
        mode="parallel",
        depends_on=[2],
        search_queries=[
            SearchQuery(
                query="career statistics of each player in the Royal Challengers Bangalore winning team",
                purpose="Gather career data for analysis"
            )
        ]
    )
    
    # Sample previous context (results from step 2)
    previous_context = """[1] Royal Challengers Bangalore team which won the IPL in 2025 had the following squad:
- Virat Kohli (Captain)
- AB de Villiers
- Chris Gayle
- KL Rahul
- Yuzvendra Chahal
- Shane Watson
- Mitchell Starc
- Iqbal Abdulla
- Harshal Patel
- Sreenath Aravind"""
    
    # Sample conversation history
    conversation_history = [
        {
            "role": "user",
            "content": "What is the career statistics of all the players who were apart of the Royal Challengers Bangalore when they won the IPL?"
        }
    ]
    
    # Test context resolution
    augmented_step = await resolver.add_context(
        current_step=current_step,
        previous_context=previous_context,
        conversation_history=conversation_history
    )

    # print(f"\n\nAugmented step: {augmented_step.model_dump_json(indent=2)}")


if __name__ == "__main__":
    asyncio.run(test_context_resolver())
