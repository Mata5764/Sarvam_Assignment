"""
Minimalistic end-to-end test script for the Deep Research Agent.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import ResearchAgent


async def test_end_to_end():
    """Test the entire research pipeline end-to-end."""
    
    # Initialize the research agent
    agent = ResearchAgent()
    
    # Test query - CHANGE THIS TO TEST DIFFERENT SCENARIOS
    #query = "What are the career statistics of all the players who were part of Royal Challengers Bangalore when they won the IPL?"
    query = "Compare the Revenue of Nvidia and AMD for the last 5 years"
    # Optional conversation history
    conversation_history = None
    
    # Run research
    print(f"Query: {query}\n")
    print("=" * 80)
    
    result = await agent.research(
        query=query,
        conversation_history=conversation_history,
        session_id="test_session"
    )
    
    print("\n" + "=" * 80)
    print("\nFINAL RESULT:")
    print(f"Answer: {result.answer.answer[:500]}..." if len(result.answer.answer) > 500 else f"Answer: {result.answer.answer}")
    print(f"\nCitations ({len(result.answer.citations)}):")
    for i, citation in enumerate(result.answer.citations[:5], 1):
        print(f"  {i}. {citation.title} - {citation.url}")


if __name__ == "__main__":
    asyncio.run(test_end_to_end())
