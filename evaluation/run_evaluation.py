"""
Evaluation runner for the Deep Research Agent.

This script runs the agent on the evaluation dataset and computes metrics.
"""
import logging
import json
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import ResearchAgent
from evaluation.dataset import get_dataset, get_categories
from evaluation.metrics import EvaluationMetrics, aggregate_results
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_evaluation(
    agent: ResearchAgent,
    dataset: list[dict],
    output_dir: Path,
    max_questions: Optional[int] = None
):
    """
    Run evaluation on the dataset.
    
    Args:
        agent: ResearchAgent instance
        dataset: List of question dicts
        output_dir: Directory to save results
        max_questions: Optional limit on number of questions
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    evaluator = EvaluationMetrics()
    results = []
    
    total = min(len(dataset), max_questions) if max_questions else len(dataset)
    
    logger.info(f"Starting evaluation on {total} questions...")
    
    for i, question_data in enumerate(dataset[:total], 1):
        question_id = question_data['id']
        question = question_data['question']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {i}/{total}: {question_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Category: {question_data['category']} | Difficulty: {question_data['difficulty']}")
        logger.info(f"{'='*80}")
        
        try:
            # Run research (async)
            logger.info("Running research...")
            result = asyncio.run(agent.research(query=question))
            
            answer = result.answer.answer
            citations = [
                {
                    'title': c.title,
                    'domain': c.domain,
                    'url': c.url
                }
                for c in result.answer.citations
            ]
            confidence = result.answer.confidence
            
            logger.info(f"\nAnswer generated ({len(answer)} chars, {len(citations)} citations)")
            logger.info(f"Confidence: {confidence}")
            
            # Evaluate
            eval_result = evaluator.evaluate(
                question_id=question_id,
                question=question,
                answer=answer,
                citations=citations,
                confidence=confidence,
                expected_elements=question_data.get('expected_elements', []),
                should_note_conflict=question_data.get('should_note_conflict', False),
                should_express_uncertainty=question_data.get('should_express_uncertainty', False)
            )
            
            # Log evaluation
            logger.info(f"\nEvaluation Results:")
            logger.info(f"  Citation Quality: {eval_result.citation_quality_score:.2f}")
            logger.info(f"  Grounding Score: {eval_result.grounding_score:.2f}")
            logger.info(f"  Completeness Score: {eval_result.completeness_score:.2f}")
            logger.info(f"  Overall Score: {eval_result.overall_score:.2f}")
            logger.info(f"  Notes: {eval_result.notes}")
            
            # Store result
            results.append({
                'question_id': question_id,
                'question': question,
                'category': question_data['category'],
                'difficulty': question_data['difficulty'],
                'answer': answer,
                'citations': citations,
                'confidence': confidence,
                'evaluation': {
                    'num_citations': eval_result.num_citations,
                    'has_citations': eval_result.has_citations,
                    'citation_format_correct': eval_result.citation_format_correct,
                    'expected_elements_found': eval_result.expected_elements_found,
                    'expected_elements_missing': eval_result.expected_elements_missing,
                    'conflict_noted': eval_result.conflict_noted,
                    'uncertainty_expressed': eval_result.uncertainty_expressed,
                    'confidence_appropriate': eval_result.confidence_appropriate,
                    'citation_quality_score': eval_result.citation_quality_score,
                    'grounding_score': eval_result.grounding_score,
                    'completeness_score': eval_result.completeness_score,
                    'overall_score': eval_result.overall_score,
                    'notes': eval_result.notes
                }
            })
            
            # Save intermediate results
            save_results(results, output_dir, aggregate_results([
                evaluator.evaluate(
                    question_id=r['question_id'],
                    question=r['question'],
                    answer=r['answer'],
                    citations=r['citations'],
                    confidence=r['confidence'],
                    expected_elements=next(
                        (q['expected_elements'] for q in dataset if q['id'] == r['question_id']),
                        []
                    ),
                    should_note_conflict=next(
                        (q.get('should_note_conflict', False) for q in dataset if q['id'] == r['question_id']),
                        False
                    ),
                    should_express_uncertainty=next(
                        (q.get('should_express_uncertainty', False) for q in dataset if q['id'] == r['question_id']),
                        False
                    )
                )
                for r in results
            ]))
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)
            results.append({
                'question_id': question_id,
                'question': question,
                'category': question_data['category'],
                'difficulty': question_data['difficulty'],
                'error': str(e),
                'evaluation': {
                    'overall_score': 0.0,
                    'notes': f"Error: {str(e)}"
                }
            })
    
    # Final aggregation
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*80}")
    
    eval_results = []
    for r in results:
        if 'error' not in r:
            q_data = next(q for q in dataset if q['id'] == r['question_id'])
            eval_results.append(
                evaluator.evaluate(
                    question_id=r['question_id'],
                    question=r['question'],
                    answer=r['answer'],
                    citations=r['citations'],
                    confidence=r['confidence'],
                    expected_elements=q_data.get('expected_elements', []),
                    should_note_conflict=q_data.get('should_note_conflict', False),
                    should_express_uncertainty=q_data.get('should_express_uncertainty', False)
                )
            )
    
    aggregated = aggregate_results(eval_results)
    
    # Save final results
    save_results(results, output_dir, aggregated)
    
    # Print summary
    print_summary(aggregated)
    
    return results, aggregated


def save_results(results: list[dict], output_dir: Path, aggregated: dict):
    """Save evaluation results to files."""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = output_dir / f"evaluation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'llm_provider': config.LLM_PROVIDER,
                'llm_model': config.LLM_MODEL,
                'search_provider': config.SEARCH_PROVIDER,
            },
            'results': results,
            'aggregated': aggregated
        }, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Save summary
    summary_file = output_dir / f"evaluation_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEEP RESEARCH AGENT - EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"LLM: {config.LLM_PROVIDER} - {config.LLM_MODEL}\n")
        f.write(f"Search: {config.SEARCH_PROVIDER}\n\n")
        f.write("="*80 + "\n")
        f.write("AGGREGATE METRICS\n")
        f.write("="*80 + "\n\n")
        
        for key, value in aggregated.items():
            if isinstance(value, float):
                f.write(f"{key:.<40} {value:.3f}\n")
            else:
                f.write(f"{key:.<40} {value}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INDIVIDUAL RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for r in results:
            f.write(f"Question ID: {r['question_id']}\n")
            f.write(f"Question: {r['question']}\n")
            if 'error' in r:
                f.write(f"ERROR: {r['error']}\n\n")
            else:
                eval_data = r['evaluation']
                f.write(f"Overall Score: {eval_data['overall_score']:.2f}\n")
                f.write(f"Citations: {eval_data['num_citations']}\n")
                f.write(f"Notes: {eval_data['notes']}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    logger.info(f"Summary saved to: {summary_file}")


def print_summary(aggregated: dict):
    """Print evaluation summary to console."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80 + "\n")
    
    print(f"Total Questions:                {aggregated['total_questions']}")
    print(f"Average Citations per Answer:   {aggregated['avg_citations']:.2f}")
    print(f"Citation Rate:                  {aggregated['citation_rate']:.1%}")
    print(f"Citation Format Correctness:    {aggregated['citation_format_rate']:.1%}")
    print(f"Completeness Rate:              {aggregated['completeness_rate']:.1%}")
    print(f"Conflict Handling Rate:         {aggregated['conflict_handling_rate']:.1%}")
    print(f"Uncertainty Handling Rate:      {aggregated['uncertainty_handling_rate']:.1%}")
    print(f"Confidence Appropriateness:     {aggregated['confidence_appropriate_rate']:.1%}")
    print()
    print(f"Average Citation Quality Score: {aggregated['avg_citation_quality']:.3f}")
    print(f"Average Grounding Score:        {aggregated['avg_grounding_score']:.3f}")
    print(f"Average Completeness Score:     {aggregated['avg_completeness_score']:.3f}")
    print(f"Average Overall Score:          {aggregated['avg_overall_score']:.3f}")
    print("\n" + "="*80 + "\n")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Run Deep Research Agent evaluation")
    parser.add_argument(
        '--max-questions',
        type=int,
        default=None,
        help='Maximum number of questions to evaluate (default: all)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Filter by category (factual, multi-hop, comparison, etc.)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results (default: ./evaluation_results)'
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    missing = config.validate()
    if missing:
        logger.error("Missing required configuration:")
        for item in missing:
            logger.error(f"  - {item}")
        logger.error("\nPlease set the required environment variables in .env file")
        sys.exit(1)
    
    # Load dataset
    dataset = get_dataset()
    
    if args.category:
        from evaluation.dataset import get_dataset_by_category
        dataset = get_dataset_by_category(args.category)
        logger.info(f"Filtered dataset to category: {args.category} ({len(dataset)} questions)")
    
    # Initialize agent
    logger.info("Initializing research agent...")
    agent = ResearchAgent()
    
    # Run evaluation
    output_dir = Path(args.output_dir)
    results, aggregated = run_evaluation(
        agent=agent,
        dataset=dataset,
        output_dir=output_dir,
        max_questions=args.max_questions
    )
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
