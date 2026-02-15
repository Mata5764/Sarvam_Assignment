"""
Main agent orchestrator that coordinates the research workflow.
Uses asyncio for parallel search and page fetching.
"""
import logging
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, field_validator, ValidationError

from agent.search import create_search_provider, SearchResult
from agent.refiner import Refiner
from agent.context_resolver import ContextResolver
from agent.answer_generator import AnswerGenerator, ResearchAnswer
from agent.strategist import Strategist
from models.strategist_models import ResearchStrategy, ExecutionStep, SearchQuery
from models.refiner_models import RefineResult
from utils.llm_client import create_llm_client
from utils.session_manager import SessionManager, TurnData
from config import config

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Complete result from a research query."""
    query: str
    answer: ResearchAnswer
    search_results: list[SearchResult] = None
    urls_used: list[str] = None


class ResearchAgent:
    """Main orchestrator for the Deep Research Agent."""
    
    def __init__(self):
        """Initialize the research agent."""
        # Get config
        llm_config = config.get_llm_config()
        search_config = config.get_search_config()
        
        # Initialize components
        self.llm_client = create_llm_client(
            provider=llm_config['provider'],
            api_key=llm_config['api_key'],
            model=llm_config['model'],
            temperature=llm_config['temperature']
        )
        self.search_provider = create_search_provider(
            provider=search_config['provider'],
            api_key=search_config['api_key']
        )
        self.refiner = Refiner()  # Creates its own LLM client
        self.context_resolver = ContextResolver()  # Creates its own LLM client
        self.answer_generator = AnswerGenerator(self.llm_client)
        self.strategist = Strategist()  # Creates its own LLM client
        self.session_manager = SessionManager()  # Session management
        
        self.max_search_results = search_config.get('max_results', 5)
        
        logger.info("ResearchAgent initialized")
    
    async def research(
        self,
        query: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
        progress_callback: Optional[Callable] = None
    ) -> ResearchResult:
        """Conduct research for a query with session management."""
        logger.info(f"Starting research for: {query}")
        start_time = datetime.now()
        
        # Handle session management
        if session_id is None:
            session_id = self.session_manager.create_session()
            logger.info(f"Created new session: {session_id}")
        elif not self.session_manager.is_session_active(session_id):
            logger.warning(f"Session {session_id} inactive, creating new session")
            session_id = self.session_manager.create_session()
        
        # Load conversation history from session if not provided
        if conversation_history is None:
            conversation_history = self.session_manager.load_conversation_history(session_id)
        
        # Save user query to conversation
        self.session_manager.save_conversation_message(session_id, "user", query)
        
        def emit(stage: str, message: str):
            if progress_callback:
                progress_callback(f"[{stage.upper()}] {message}")
        
        try:
            # Step 0: Plan research strategy
            emit("plan", "Planning research strategy...")
            strategy = await self.strategist.plan_research_strategy(query, conversation_history)
            emit("plan", f"Strategy: {strategy.execution_type} with {len(strategy.steps)} steps - {strategy.reason_summary}")
            
            # Step 1 & 2: Execute steps to get refined data
            all_refined_data, all_search_queries, urls_used = await self._execute_steps(
                strategy=strategy,
                original_query=query,
                conversation_history=conversation_history,
                emit=emit
            )
            
            if not all_refined_data:
                result = self._create_no_results_response(query)
                self._save_turn_to_session(
                    session_id, query, strategy.execution_type, all_search_queries,
                    [], all_refined_data, result.answer, start_time
                )
                return result
            
            # Step 3: Generate answer from refined data
            emit("answer", "Generating answer from refined data...")
            answer = await self.answer_generator.generate_answer(query, all_refined_data)
            emit("answer", f"Complete! ({len(answer.citations)} citations)")
            
            # Save assistant answer to conversation
            self.session_manager.save_conversation_message(session_id, "assistant", answer.answer)
            
            # Create result
            result = ResearchResult(
                query=query,
                answer=answer,
                search_results=[],
                urls_used=list(set(urls_used))
            )
            
            # Save turn history
            self._save_turn_to_session(
                session_id, query, strategy.execution_type, all_search_queries,
                result.urls_used, all_refined_data, answer, start_time
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            result = ResearchResult(
                query=query,
                answer=ResearchAnswer(
                    answer=f"I encountered an error during research: {str(e)}",
                    citations=[],
                    confidence="low",
                    conflicts_detected=False
                ),
                search_results=[],
                urls_used=[]
            )
            
            # Save error turn
            self._save_turn_to_session(
                session_id, query, "error", [], [], [], result.answer, start_time
            )
            
            return result
    
    async def _execute_steps(
        self,
        strategy: ResearchStrategy,
        original_query: str,
        conversation_history: Optional[list[dict]],
        emit: Callable
    ) -> tuple[list[dict], list[str], list[str]]:
        """
        Execute strategy steps in order, handling dependencies and actions.
        Returns: (all_refined_data, all_search_queries, urls_used)
        """
        all_refined_data = []
        all_search_queries = []
        urls_used = []
        step_results = {}  # Store results by step_id for dependency resolution
        
        for step in strategy.steps:
            emit("step", f"Step {step.step_id}: {step.description}")
            logger.info(f"Executing step {step.step_id} (action={step.action}, mode={step.mode})")
            
            # Check dependencies
            if step.depends_on:
                missing_deps = [dep_id for dep_id in step.depends_on if dep_id not in step_results]
                if missing_deps:
                    logger.error(f"Step {step.step_id} has missing dependencies: {missing_deps}")
                    continue
            
            # Handle based on action type
            if step.action == "search":
                # Refine queries if step has dependencies (use previous results)
                if step.depends_on:
                    emit("context", f"Refining queries for step {step.step_id} based on previous results...")
                    
                    # Collect previous refined data from dependencies
                    previous_data = []
                    for dep_id in step.depends_on:
                        if dep_id in step_results:
                            previous_data.extend(step_results[dep_id])
                    
                    # Use ContextResolver to generate context-aware queries
                    refined_queries = await self.context_resolver.refine_queries(
                        current_step=step,
                        previous_results=previous_data,
                        conversation_history=conversation_history,
                        original_query=original_query
                    )
                    
                    # Update step's search queries with refined ones
                    step.search_queries = [SearchQuery(query=q) for q in refined_queries]
                    logger.info(f"Refined {len(refined_queries)} queries for step {step.step_id}")
                
                # Execute searches
                step_refined_data = await self._execute_search_step(
                    step=step,
                    original_query=original_query,
                    conversation_history=conversation_history,
                    previous_results=step_results,
                    emit=emit
                )
                
                # Collect refined data with query tracking
                # step_refined_data maintains order corresponding to step.search_queries
                for idx, search_query in enumerate(step.search_queries):
                    all_search_queries.append(search_query.query)
                    
                    # Get corresponding refined data (now always a dict, never None)
                    if idx < len(step_refined_data):
                        data = step_refined_data[idx]
                        
                        # Only add if search was successful (score > 0)
                        if data.get("score", 0) > 0:
                            all_refined_data.append(data)
                            
                            # Extract URLs from refined data
                            if "sources" in data:
                                urls_used.extend([s.get("url", "") for s in data["sources"]])
                        else:
                            # Log failed searches but don't add to refined_data
                            logger.info(f"Skipping failed search result for query: {search_query.query}")
                
                # Store results for this step (only successful ones with score > 0)
                successful_results = [d for d in step_refined_data if d.get("score", 0) > 0]
                step_results[step.step_id] = successful_results
                emit("step", f"Step {step.step_id} completed: {len(successful_results)}/{len(step.search_queries)} successful")
            
            elif step.action == "generation":
                # Generation step - just prepare data for answer generation
                emit("step", f"Step {step.step_id}: Ready for answer generation")
                # Collect data from dependencies
                for dep_id in step.depends_on:
                    if dep_id in step_results:
                        # Already added to all_refined_data
                        pass
                step_results[step.step_id] = []  # Mark as completed
            
            else:
                logger.warning(f"Unknown action type: {step.action}")
        
        return all_refined_data, all_search_queries, urls_used
    
    async def _execute_search_step(
        self,
        step: ExecutionStep,
        original_query: str,
        conversation_history: Optional[list[dict]],
        previous_results: dict,
        emit: Callable
    ) -> list[dict]:
        """
        Execute a single search step (can be single or parallel mode).
        
        Returns: List of refined data in same order as step.search_queries.
                 None entries for failed queries to maintain order mapping.
        """
        if not step.search_queries:
            logger.warning(f"Step {step.step_id} has no search queries")
            return []
        
        try:
            if step.mode == "single":
                # Execute single query
                search_query = step.search_queries[0]
                emit("search", f"Searching: {search_query.query}")
                refined_data = await self._search_with_refine(search_query.query, emit)
                return [refined_data]  # Return as list with one element (always a dict)
            
            elif step.mode == "parallel":
                # Execute queries in parallel
                emit("search", f"Searching {len(step.search_queries)} queries in parallel...")
                tasks = [
                    self._search_with_refine(
                        search_query.query,
                        lambda stage, msg, idx=idx: emit("search", f"[{idx}/{len(step.search_queries)}] {msg}")
                    )
                    for idx, search_query in enumerate(step.search_queries, 1)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results, maintaining order
                refined_data_list = []
                for idx, result in enumerate(results, 1):
                    if isinstance(result, dict):
                        refined_data_list.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Query {idx} failed with exception: {result}")
                        # Create error result dict for consistency
                        refined_data_list.append({
                            "refined_data": f"Search failed due to exception",
                            "sources": [],
                            "query": step.search_queries[idx-1].query,
                            "score": 0.0,
                            "error": str(result)
                        })
                    else:
                        logger.warning(f"Query {idx} returned unexpected type: {type(result)}")
                        refined_data_list.append({
                            "refined_data": f"Search returned unexpected result",
                            "sources": [],
                            "query": step.search_queries[idx-1].query,
                            "score": 0.0,
                            "error": f"Unexpected type: {type(result)}"
                        })
                
                return refined_data_list  # Returns list with same length as search_queries
            
            else:
                logger.error(f"Unknown mode: {step.mode}")
                return []
        
        except Exception as e:
            logger.error(f"Search step {step.step_id} failed: {e}", exc_info=True)
            return []
    
    async def _search_with_refine(self, query: str, emit: Callable) -> dict:
        """
        Execute search with refiner - handles retries if needed.
        
        Returns: Dict with refined data and source URLs:
                 {
                     "refined_data": str,  # Extracted relevant info
                     "sources": [{"url": str, "title": str}, ...],
                     "query": str,
                     "score": float,
                     "error": str (optional)  # Present if all retries failed
                 }
                 Always returns a dict (never None). Score 0.0 indicates total failure.
        """
        
        retry_count = 0
        max_retries = 3
        best_result = None  # Track best result across retries
        
        while retry_count <= max_retries:
            # Execute search
            emit("search", f"Searching: '{query}' (attempt {retry_count + 1})")
            search_results = await self.search_provider.search(query, self.max_search_results)
            
            # Validate search results
            validation_result = self._validate_search_results(search_results, query, retry_count, max_retries, emit)
            if validation_result is None:
                # Validation failed, retry
                retry_count += 1
                continue
            
            results_with_content = validation_result
            
            # Refine results (only quality validation now)
            emit("refine", "Validating and extracting relevant info...")
            refine_result = await self.refiner.refine_search_results(
                query=query,
                results=results_with_content,  # Pass filtered results
                retry_count=retry_count
            )
            
            emit("refine", f"Quality score: {refine_result.score:.2f} - {refine_result.reason}")
            
            # Build result structure with only used sources
            current_result = {
                "refined_data": refine_result.refined_data,
                "sources": [
                    {
                        "url": results_with_content[i].url,
                        "title": results_with_content[i].title,
                        "domain": results_with_content[i].url.split('/')[2] if '/' in results_with_content[i].url else results_with_content[i].url
                    }
                    for i in refine_result.source_indices
                    if i < len(results_with_content)  # Bounds check
                ],
                "query": query,
                "score": refine_result.score
            }
            
            # Track best result so far
            if best_result is None or current_result["score"] > best_result["score"]:
                best_result = current_result
            
            # Check if we should retry
            if refine_result.should_retry:
                emit("refine", f"Low quality results, retrying...")
                retry_count += 1
                continue
            else:
                # Success - return current result
                return current_result
        
        # Max retries reached
        if best_result:
            # Return best attempt we found
            logger.warning(f"Max retries reached for query '{query}', using best result (score: {best_result['score']})")
            emit("refine", f"Max retries reached, using best result (score: {best_result['score']:.2f})")
            return best_result
        else:
            # All retries failed - return structured failure result
            logger.error(f"All retries failed for query '{query}', no valid results found")
            emit("refine", "All retries failed, no valid results found")
            return {
                "refined_data": f"Unable to find relevant information for query: {query}",
                "sources": [],
                "query": query,
                "score": 0.0,
                "error": "All search attempts failed (empty results or no substantial content)"
            }
    
    def _validate_search_results(
        self,
        search_results: list,
        query: str,
        retry_count: int,
        max_retries: int,
        emit: Callable
    ) -> Optional[list]:
        """
        Validate search results have substantial content.
        
        Returns:
            List of results with content if valid, None if validation fails
        """
        # Check for empty results
        if not search_results or len(search_results) == 0:
            emit("search", "No results found, retrying..." if retry_count < max_retries else "No results found")
            logger.warning(f"No search results for query '{query}' (attempt {retry_count + 1})")
            return None
        
        # Filter results with substantial content
        results_with_content = [r for r in search_results if r.content and len(r.content.strip()) > 50]
        
        if not results_with_content:
            emit("search", "No substantial content, retrying..." if retry_count < max_retries else "No substantial content")
            logger.warning(f"No substantial content in search results for query '{query}' (attempt {retry_count + 1})")
            return None
        
        emit("search", f"Found {len(results_with_content)} results with content")
        return results_with_content
    
    def _create_no_results_response(self, query: str) -> ResearchResult:
        """Create response when no search results are found."""
        from agent.answer_generator import ResearchAnswer
        
        return ResearchResult(
            query=query,
            answer=ResearchAnswer(
                answer=f"No search results found for: '{query}'. Please try rephrasing your question.",
                citations=[],
                confidence="low",
                conflicts_detected=False
            )
        )
    
    def _save_turn_to_session(
        self,
        session_id: str,
        query: str,
        strategy: str,
        search_queries: list[str],
        urls_opened: list[str],
        refined_data: list[dict],
        answer: ResearchAnswer,
        start_time: datetime
    ):
        """Save turn data to session."""
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        turn_count = self.session_manager.get_turn_count(session_id)
        
        turn_data = TurnData(
            turn_id=turn_count + 1,
            query=query,
            strategy=strategy,
            search_queries=search_queries,
            urls_opened=urls_opened,
            refined_data=refined_data,
            final_answer=answer.answer,
            citations=[{"title": c.title, "domain": c.domain, "url": c.url} for c in answer.citations],
            timestamp=start_time.isoformat(),
            duration_ms=duration_ms
        )
        
        self.session_manager.save_turn(session_id, turn_data)
