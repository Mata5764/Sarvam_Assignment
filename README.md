# 🔍 Deep Research Agent

A web research agent that conducts multi-step research, validates sources, and provides citation-grounded answers. Built from scratch without frameworks.

## Features

- Intelligent research planning (single vs. multi-step strategies)
- Context-aware sequential searches
- Quality retrieval with automatic retries
- Citation-grounded answers with source tracking
- Session management with conversation history
- Async architecture for parallel searches
- Multi-LLM provider support

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up .env file like below as per your model choice (Use "openai" for GPT, "anthropic" for claude, "google" for gemini and  "groq" for Groq
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly_...

STRATEGIST_LLM_PROVIDER=groq
STRATEGIST_LLM_MODEL=llama-3.3-70b-versatile

REFINER_LLM_PROVIDER=groq
REFINER_LLM_MODEL=llama-3.3-70b-versatile

CONTEXT_RESOLVER_LLM_PROVIDER=groq
CONTEXT_RESOLVER_LLM_MODEL=llama-3.3-70b-versatile

ANSWER_GENERATOR_LLM_PROVIDER=groq
ANSWER_GENERATOR_LLM_MODEL=llama-3.3-70b-versatile

LLM_JUDGE_PROVIDER=groq
LLM_JUDGE_MODEL=moonshotai/kimi-k2-instruct-0905

# 3. Run
streamlit run app.py
```

## Architecture

```
                    ┌──────────────────┐
                    │   User Query     │
                    └────────┬─────────┘
                             ↓
                    ┌────────────────┐
                    │  Strategist    │ (single or chain)
                    └────────┬───────┘
                             ↓
        ╔════════════════════════════════════════╗
        ║         For Each Step:                 ║
        ║                                        ║
        ║    ┌──────────────────┐                ║
        ║    │ Context Resolver │ ←─ (previous)  ║
        ║    └────────┬─────────┘                ║
        ║             ↓                          ║
        ║    ┌──────────────────┐                ║
        ║    │     Search       │←─------┐       ║
        ║    └────────┬─────────┘        |       ║
        ║             ↓                  |       ║
        ║    ┌──────────────────┐        |       ║
        ║    │    Refiner       │        |       ║
        ║    │  (score/extract) │        |       ║
        ║    └────────┬─────────┘        |       ║
        ║             ↓                  |       ║
        ║        [score low?] ──yes── (retry)    ║
        ║             │                          ║
        ║            no                          ║
        ║             ↓                          ║
        ║      [store refined]                   ║
        ║                                        ║
        ╚════════════════════════════════════════╝
                             ↓
                    ┌────────────────┐
                    │Answer Generator│
                    └────────┬───────┘
                             ↓
                    ┌────────────────┐
                    │Session Storage │
                    └────────────────┘
```

**Components:**
- `Strategist`: Plans single/chain research strategy
- `Search`: Tavily API with content extraction
- `Refiner`: Validates quality, scores relevance, refines content and retries if needed
- `Context Resolver`: Refines queries based on previous results
- `Answer Generator`: Generates final response and Synthesizes citations

## Design Note

### Target Users & Problem

**Users:** Researchers, students, analysts needing cited information from multiple sources.

**Problem:** Manual research requires visiting links, reading, synthesizing, and tracking citations. This agent automates the workflow.

### What is "Deep Research"?

1. **Intelligent Query Planning**: Auto-determines single vs. multi-step research
2. **Context-Aware Search**: Refines queries based on previous results
3. **Quality-Gated Retrieval**: Validates with scoring and auto-retries
4. **Multi-Source Synthesis**: Combines sources with exact attribution
5. **Source Transparency**: Full traceability from raw content to citations

### Success Metrics (5)

- **Source Attribution >0.90**: Claims must be backed by citations
- **Search Quality Score >0.75**: Relevance and completeness of sources
- **Strategy Appropriateness >0.85**: Correct single vs. chain decision
- **Answer Completeness >0.70**: Coverage of all query aspects
- **Source Diversity ≥3**: Unique domains to avoid echo chambers

### Data Flow

```
User Query
    ↓
[Strategist] single/chain decision
    ↓
For Each Step:
    [Context Resolver] refine queries (if chained)
    [Search] Tavily → content
    [Refiner] score → extract → retry if low quality
    ↓
[Answer Generator] synthesize + citations
    ↓
[Session] store conversation + turns + raw content
```

### Risks & Limitations

1. **Search Quality**: Depends on Tavily results
2. **Context Windows**: Max ~16K tokens per research
3. **Hallucination**: LLM may add unsupported claims

### Future Improvements

**1. Adaptive Context Management**
- Recursive summarization for long docs

**2. Better Prompts to reduce hallucination**
- Addition of few shoots

**2. Better Search**
- Try out Parallel or other search alternatives

## Evaluation

LLM-as-Judge evaluation that scores each step of the research process.

```bash
# Full evaluation (12 questions)
python3 evaluation/run_evaluation.py

# Quick test (1 question)
python3 evaluation/run_evaluation.py --max-questions 1

# Specific category
python3 evaluation/run_evaluation.py --max-questions 5
```

**Evaluation Criteria:**

The LLM Judge evaluates 5 components with chain-of-thought reasoning:

1. **Strategy Planning** (20%): Execution type, step logic, query formulation
2. **Web Search Quality** (20%): Query relevance, result quality, information utility
3. **Content Refinement** (20%): Extraction accuracy, refiner scores, relevance
4. **Context Resolution** (10%): Query enhancement, specificity, appropriateness
5. **Answer Generation** (30%): Completeness, citation quality, accuracy, clarity

**Outputs**:
- Per-question scores (0.0-1.0) with detailed reasoning for each component
- JSON results with aggregate metrics
- Console summary with weighted overall scores

See `evaluation/LLM_JUDGE.md` for full details.

## Project Structure

```
agent/
  ├── orchestrator.py       # Main coordinator
  ├── strategist.py         # Research planner
  ├── search.py             # Tavily integration
  ├── refiner.py            # Quality validation
  ├── context_resolver.py   # Query refinement
  └── answer_generator.py   # Synthesis

models/                     # Pydantic schemas
prompts/                    # Structured prompts
utils/                      # LLM client, session manager
evaluation/                 # Test harness
```

## Requirements Compliance

✅ No frameworks (pure Python)  
✅ Web research (Tavily)  
✅ Citations with URLs  
✅ Session management (JSON)  
✅ Streaming progress  
✅ Design note (above)  
✅ Evaluation harness  
✅ Streamlit UI 
