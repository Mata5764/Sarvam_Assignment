"""
Evaluation runner using LLM-as-Judge.

Runs the agent on dataset questions and evaluates with LLM judge.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.orchestrator import ResearchAgent
from evaluation.dataset import get_dataset, get_dataset_by_category
from evaluation.llm_judge import LLMJudge, format_judge_result
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_evaluation(
    dataset: list[dict],
    output_dir: Path,
    max_questions: int = None
):
    """Run LLM judge evaluation on dataset."""
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    agent = ResearchAgent()
    judge = LLMJudge()
    
    results = []
    total = min(len(dataset), max_questions) if max_questions else len(dataset)
    
    logger.info(f"Starting evaluation on {total} questions...\n")
    
    for i, question_data in enumerate(dataset[:total], 1):
        question_id = question_data['id']
        question = question_data['question']
        
        logger.info(f"{'='*80}")
        logger.info(f"Question {i}/{total}: {question}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Run research
            result = await agent.research(query=question)
            
            # Get actual strategy and search data from orchestrator
            strategy_data = result.strategy_data
            search_steps_data = result.search_steps_data
            
            citations = [
                {'title': c.title, 'domain': c.domain, 'url': c.url}
                for c in result.answer.citations
            ]
            
            # Evaluate with LLM judge (single call)
            judge_result = await judge.evaluate(
                question_id=question_id,
                question=question,
                strategy_data=strategy_data,
                search_steps_data=search_steps_data,
                answer=result.answer.answer,
                citations=citations
            )
            
            # Print result
            print(f"\n{format_judge_result(judge_result)}\n")
            
            # Store
            results.append({
                'question_id': question_id,
                'question': question,
                'category': question_data['category'],
                'answer': result.answer.answer,
                'citations': citations,
                'evaluation': {
                    'strategy_score': judge_result.strategy_score,
                    'strategy_reasoning': judge_result.strategy_reasoning,
                    'avg_search_score': judge_result.avg_search_score,
                    'answer_score': judge_result.answer_score,
                    'answer_reasoning': judge_result.answer_reasoning,
                    'overall_score': judge_result.overall_score
                }
            })
            
        except Exception as e:
            logger.error(f"Error: {e}")
            results.append({
                'question_id': question_id,
                'question': question,
                'error': str(e)
            })
    
    # Save results
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"evaluation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_questions': total,
            'avg_overall_score': sum(r['evaluation']['overall_score'] for r in results if 'evaluation' in r) / len([r for r in results if 'evaluation' in r]),
            'results': results
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    successful = [r for r in results if 'evaluation' in r]
    print(f"Total: {total}")
    print(f"Successful: {len(successful)}")
    print(f"Avg Overall Score: {sum(r['evaluation']['overall_score'] for r in successful) / len(successful):.2f}")
    print(f"Avg Strategy Score: {sum(r['evaluation']['strategy_score'] for r in successful) / len(successful):.2f}")
    print(f"Avg Search Score: {sum(r['evaluation']['avg_search_score'] for r in successful) / len(successful):.2f}")
    print(f"Avg Answer Score: {sum(r['evaluation']['answer_score'] for r in successful) / len(successful):.2f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run LLM Judge evaluation")
    parser.add_argument('--max-questions', type=int, default=None)
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')
    
    args = parser.parse_args()
    
    # Validate config
    missing = config.validate()
    if missing:
        logger.error("Missing configuration:")
        for item in missing:
            logger.error(f"  - {item}")
        sys.exit(1)
    
    # Load dataset
    dataset = get_dataset()
    if args.category:
        dataset = get_dataset_by_category(args.category)
        logger.info(f"Filtered to category: {args.category} ({len(dataset)} questions)")
    
    # Run
    output_dir = Path(args.output_dir)
    asyncio.run(run_evaluation(dataset, output_dir, args.max_questions))


if __name__ == "__main__":
    main()
