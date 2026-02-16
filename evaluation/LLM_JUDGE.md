# LLM-as-Judge Evaluation

A comprehensive evaluation system using an LLM to assess the quality of the Deep Research Agent's complete workflow.

## Overview

The LLM Judge evaluates the entire research process in a single comprehensive call, scoring 5 key components:

1. **Strategy Planning** (20%)
2. **Web Search Quality** (20%)
3. **Content Refinement** (20%)
4. **Context Resolution** (10%)
5. **Answer Generation** (30%)

## Evaluation Criteria

### 1. Strategy Planning
- Was the execution type (single/chain) appropriate for the question?
- Were the steps logical, complete, and well-structured?
- Were search queries well-formulated and relevant?

### 2. Web Search Quality
- Were search queries relevant and specific to information needs?
- Did searches return useful, on-topic information?
- Quality across all search steps

### 3. Content Refinement
- Did the refiner successfully extract relevant information?
- Were refiner scores accurate indicators of quality?
- Was extracted information concise, relevant, and on-topic?

### 4. Context Resolution
- Were queries meaningfully improved with previous context?
- Did refined queries add specificity using concrete information?
- Was it appropriate when skipped (no dependencies)?

### 5. Answer Generation
- Is the answer accurate, complete, and addresses the question?
- Are citations properly used, relevant, and from quality sources?
- Is the answer well-structured and understandable?

## Prompt Structure

The LLM Judge uses structured XML tags for chain-of-thought reasoning:

```xml
<strategy_evaluation>
[Natural reasoning about strategy quality]
</strategy_evaluation>

<search_evaluation>
[Natural reasoning about search quality]
</search_evaluation>

<refinement_evaluation>
[Natural reasoning about refinement quality]
</refinement_evaluation>

<context_resolution_evaluation>
[Natural reasoning about context resolution, or "N/A - not used"]
</context_resolution_evaluation>

<answer_evaluation>
[Natural reasoning about answer quality]
</answer_evaluation>

<overall_assessment>
[Summary of overall research quality]
</overall_assessment>

<response>
{
  "strategy_score": 0.0,
  "search_score": 0.0,
  "refinement_score": 0.0,
  "context_score": 0.0,
  "answer_score": 0.0,
  "overall_score": 0.0
}
</response>
```

## Usage

### Individual Evaluation

```python
from evaluation.llm_judge import LLMJudge

judge = LLMJudge()

result = await judge.evaluate(
    question_id="query_1",
    question="What are the main differences between GPT-4 and Claude?",
    strategy_data=agent_result.strategy_data,
    search_steps_data=agent_result.search_steps_data,
    answer=agent_result.answer.answer,
    citations=[{"title": c.title, "url": c.url, "domain": c.domain} 
               for c in agent_result.answer.citations]
)

print(f"Overall Score: {result.overall_score:.2f}")
print(f"Strategy: {result.strategy_score:.2f}")
print(f"Search: {result.search_score:.2f}")
print(f"Refinement: {result.refinement_score:.2f}")
print(f"Context: {result.context_score:.2f}")
print(f"Answer: {result.answer_score:.2f}")
```

### Batch Evaluation

```bash
# Evaluate on dataset
python evaluation/run_evaluation.py --max-questions 5

# Evaluate all questions
python evaluation/run_evaluation.py
```

## Output Format

```
COMPONENT EVALUATIONS
======================================================================

1. Strategy Planning: 0.75/1.0
   The chain execution was appropriate for this comparative question...

2. Web Search Quality: 0.80/1.0
   Search queries were relevant and specific to information needs...

3. Content Refinement: 0.85/1.0
   Refiner successfully extracted relevant information with high scores...

4. Context Resolution: 0.70/1.0
   Queries were meaningfully improved with previous context...

5. Answer Generation: 0.90/1.0
   Answer is accurate, complete, well-cited, and addresses the question...

======================================================================
OVERALL SCORE: 0.82/1.0
======================================================================
Weights: Strategy 20% | Search 20% | Refinement 20% | Context 10% | Answer 30%
```

## Configuration

The LLM Judge uses a dedicated LLM client. Configure in `.env`:

```bash
# LLM Judge Configuration
LLM_JUDGE_PROVIDER=groq
LLM_JUDGE_MODEL=moonshotai/kimi-k2-instruct-0905
LLM_JUDGE_TEMPERATURE=0.3
```

## Evaluation Results

Results are saved to `evaluation/results/YYYY-MM-DD_HH-MM-SS_results.json`:

```json
{
  "timestamp": "2026-02-16T10:30:00",
  "total_questions": 5,
  "results": [
    {
      "question_id": "query_1",
      "question": "...",
      "overall_score": 0.82,
      "strategy_score": 0.75,
      "search_score": 0.80,
      "refinement_score": 0.85,
      "context_score": 0.70,
      "answer_score": 0.90,
      "reasoning": {...}
    }
  ],
  "aggregate": {
    "mean_overall": 0.82,
    "mean_strategy": 0.75,
    "mean_search": 0.80,
    "mean_refinement": 0.85,
    "mean_context": 0.70,
    "mean_answer": 0.90
  }
}
```

## Design Philosophy

### Single Comprehensive Call
Instead of multiple separate evaluations, one LLM call sees the complete workflow, enabling:
- Holistic assessment of the research process
- Understanding of dependencies between components
- Consistent evaluation across all dimensions
- Cost and latency efficiency

### Chain-of-Thought Reasoning
Each component has a dedicated reasoning section where the LLM explains its assessment before assigning scores, ensuring:
- Transparent evaluation logic
- Detailed feedback for improvement
- Consistent scoring criteria

### Structured Output
Scores are provided in JSON format at the end, enabling:
- Easy parsing and aggregation
- Programmatic analysis
- Clear separation of reasoning and scores
