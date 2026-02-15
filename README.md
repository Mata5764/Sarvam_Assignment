# 🔍 Deep Research Agent

A web research agent that searches the internet, retrieves content from multiple sources, and provides citation-grounded answers. Built from scratch without using LangChain, LangGraph, or similar frameworks.

## Features

- **Web Research**: Searches via Tavily/Serper APIs
- **Citation-Grounded Answers**: Every claim backed by `[Title — domain](URL)` citations
- **Session Management**: Persistent conversation history in SQLite
- **Conflict Detection**: Explicitly notes when sources disagree
- **Streamlit UI**: Clean chat interface with progress updates
- **Multi-Provider Support**: Works with OpenAI, Anthropic, Google Gemini, or Groq

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file:

```bash
# Choose your LLM provider (pick one)
OPENAI_API_KEY=sk-...           # OpenAI
GROQ_API_KEY=gsk_...           # Groq (free tier available)
GOOGLE_API_KEY=...             # Google Gemini
ANTHROPIC_API_KEY=sk-ant-...   # Anthropic Claude

# Search provider
TAVILY_API_KEY=tvly-...        # Recommended: 1000 free searches/month

# Configuration
LLM_PROVIDER=groq                    # or: openai, google, anthropic
LLM_MODEL=llama-3.3-70b-versatile   # or: gpt-4o-mini, gemini-2.0-flash-exp
SEARCH_PROVIDER=tavily
```

**Recommended Free Setup:**
- LLM: Groq (free, fast) - https://console.groq.com/
- Search: Tavily (1000 free/month) - https://tavily.com/

### 3. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 and start researching!

## Programmatic Usage

```python
import asyncio
from agent import ResearchAgent

# Initialize agent
agent = ResearchAgent()

# Ask a question (async)
async def research():
    result = await agent.research("What are the latest developments in quantum computing?")
    
    # Access answer and citations
    print(result.answer.answer)
    print(f"Confidence: {result.answer.confidence}")
    
    for citation in result.answer.citations:
        print(f"  [{citation.title}]({citation.url})")

# Run
asyncio.run(research())
```

## Architecture

```
User Query
    ↓
Plan Research (generate 2-3 search queries)
    ↓
Search Web (Tavily/Serper)
    ↓
Fetch Pages (top 5 URLs)
    ↓
Extract & Select Context (relevant snippets)
    ↓
Generate Answer (with citations)
    ↓
Store in Session (SQLite)
```

### Components

- **`agent/orchestrator.py`**: Main research flow coordinator
- **`agent/search.py`**: Web search interface (Tavily/Serper)
- **`agent/fetcher.py`**: Page content retrieval
- **`agent/context_builder.py`**: Snippet selection & ranking
- **`agent/answer_generator.py`**: Citation-grounded answer generation
- **`storage/`**: SQLite session management
- **`utils/llm_client.py`**: Multi-provider LLM interface
- **`app.py`**: Streamlit UI

## Design Note

### Problem & Users

**Target Users:** Researchers, students, analysts, and knowledge workers who need comprehensive, cited information from multiple sources.

**Problem Solved:** Traditional search returns links but requires manual visiting, reading, synthesizing, and citation tracking. This agent automates the entire research workflow.

### What is "Deep Research"?

For this implementation, deep research means:

1. **Multi-Source Investigation**: Generating diverse search queries and fetching content from multiple URLs
2. **Intelligent Context Selection**: Prioritizing relevant, recent, and diverse information sources
3. **Synthesis with Attribution**: Combining information while maintaining citation integrity
4. **Conflict Detection**: Identifying and explicitly noting when sources disagree
5. **Uncertainty Expression**: Clearly stating when evidence is insufficient

### Success Metrics

I chose **5 key metrics** to measure research quality:

#### 1. Citation Quality Score (0-1)
- **Measures**: Proper source attribution and formatting
- **Why**: Foundation of research credibility
- **Calculation**: (Number of citations × 0.6) + (Format correctness × 0.4)

#### 2. Grounding Score (0-1)
- **Measures**: Answer grounded in sources vs. generated
- **Why**: Prevents hallucination
- **Calculation**: Binary - has properly formatted citations or not

#### 3. Completeness Score (0-1)
- **Measures**: Coverage of expected answer elements
- **Why**: Ensures thorough answers
- **Calculation**: Found elements / Total expected elements

#### 4. Conflict Handling Rate (%)
- **Measures**: Detection of source disagreements
- **Why**: Critical for controversial topics
- **Calculation**: Conflicts noted / Conflicts present

#### 5. Uncertainty Expression Rate (%)
- **Measures**: Appropriate uncertainty communication
- **Why**: Intellectual honesty and trust
- **Calculation**: Uncertainty expressed / Insufficient evidence cases

### Data Flow

```
User Query → Generate Search Queries → Search Web → Fetch Pages
     ↓
Extract Text → Score & Rank Sources → Select Snippets (within token budget)
     ↓
Build Prompt → Generate Answer → Parse Citations → Return Result
     ↓
Store: Session DB (conversation history + research turns)
```

### Risks & Limitations

1. **Rate Limits**: API rate limits can throttle research (mitigated by provider selection)
2. **Source Quality**: No credibility scoring (future: integrate fact-checker APIs)
3. **Context Limits**: Fixed 8000 token budget (future: hierarchical summarization)
4. **Language**: English-only support
5. **Media**: Text-only (no PDFs, images, videos)

### Future Improvements

**1. Source Credibility Scoring**
- Maintain database of trusted sources (academic journals, fact-checkers)
- Cross-reference claims across multiple sources
- Provide credibility ratings in citations

**2. Iterative Research**
- Track information gaps
- Generate follow-up searches
- Multi-hop reasoning across research turns

## Evaluation

### Running Evaluation

```bash
# Full evaluation (12 questions)
python evaluation/run_evaluation.py

# Specific category
python evaluation/run_evaluation.py --category factual

# Quick test
python evaluation/run_evaluation.py --max-questions 3
```

### Dataset

12 diverse questions across 6 categories:
- **Factual** (3): Simple fact retrieval
- **Multi-hop** (3): Connect info from multiple sources  
- **Comparison** (2): Compare two entities
- **Recent** (1): Test currency of information
- **Technical** (1): Complex explanations
- **Conflicting** (1): Sources with disagreements
- **Insufficient** (1): Limited available information

### Expected Performance

| Metric | Target | 
|--------|--------|
| Citation Rate | >90% |
| Grounding Score | >0.8 |
| Completeness | >0.6 |
| Conflict Handling | >80% |
| Uncertainty Expression | >90% |

## Project Structure

```
Sarvam_Assignment/
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── config.py                   # Configuration
├── app.py                      # Streamlit UI
│
├── agent/                      # Core agent
│   ├── orchestrator.py         # Main research loop
│   ├── search.py               # Web search
│   ├── fetcher.py              # Page fetching
│   ├── context_builder.py      # Context selection
│   └── answer_generator.py     # Answer generation
│
├── storage/                    # Session persistence
│   ├── database.py             # SQLite interface
│   └── session_manager.py      # Session management
│
├── utils/                      # Utilities
│   ├── llm_client.py           # LLM providers
│   └── text_processing.py      # Text extraction
│
└── evaluation/                 # Evaluation harness
    ├── dataset.py              # Test questions
    ├── metrics.py              # Evaluation metrics
    └── run_evaluation.py       # Runner script
```

## Example Conversation

**User:** What is the capital of France and its population?

**Agent:** The capital of France is Paris. According to recent estimates, Paris has a population of approximately 2.16 million people within its city limits [Paris — wikipedia.org](https://en.wikipedia.org/wiki/Paris), while the Paris metropolitan area has around 12.4 million people [Île-de-France — wikipedia.org](https://en.wikipedia.org/wiki/%C3%8Ele-de-France).

**Confidence:** high

---

**User:** What is the effectiveness of vitamin D supplementation in preventing COVID-19?

**Agent:** The effectiveness of vitamin D supplementation in preventing COVID-19 shows conflicting evidence. Some studies suggest that vitamin D deficiency is associated with increased COVID-19 severity [NIH Study — nih.gov](https://...), while other research indicates that supplementation does not significantly reduce infection risk [NEJM Study — nejm.org](https://...). This disagreement suggests more research is needed for definitive conclusions.

**Confidence:** medium
⚠️ Sources present differing information

## Configuration

All configuration via environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider | `openai` |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `SEARCH_PROVIDER` | Search provider | `tavily` |
| `MAX_SEARCH_RESULTS` | Results per query | `5` |
| `MAX_PAGES_TO_FETCH` | Pages to fetch | `5` |
| `MAX_CONTEXT_TOKENS` | Max context size | `8000` |

## Tech Stack

- **Python 3.10+**
- **Streamlit** - UI
- **SQLite** - Session storage
- **Tavily/Serper** - Web search
- **OpenAI/Anthropic/Google/Groq** - LLMs
- **Trafilatura + BeautifulSoup** - Content extraction
- **tiktoken** - Token counting

## Requirements Compliance

✅ **No Frameworks**: Pure Python implementation, no LangChain/LangGraph/CrewAI  
✅ **Web Research**: Tavily/Serper integration  
✅ **Citations**: `[Title — domain](URL)` format  
✅ **Session Management**: SQLite with conversation + turn history  
✅ **Streaming Progress**: Real-time updates for all stages  
✅ **Design Note**: Complete in README  
✅ **Evaluation Harness**: 12 questions, 5 metrics, runnable script  
✅ **Working UI**: Streamlit app  

## License

MIT

## Acknowledgments

Built for the Sarvam AI Agent Challenge. Thanks to Tavily for their excellent search API.
