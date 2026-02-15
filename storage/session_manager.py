"""
Session manager for handling conversation and research state.
"""
import logging
import uuid
from typing import Optional
from datetime import datetime
from storage.database import Database
from config import config

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages research sessions with conversation and turn history."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize session manager.
        
        Args:
            db_path: Optional path to database file
        """
        self.db = Database(db_path or config.DATABASE_PATH)
    
    def create_session(self, session_id: Optional[str] = None, metadata: Optional[dict] = None) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Optional session ID (generates UUID if not provided)
            metadata: Optional metadata dictionary
            
        Returns:
            Session ID
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self.db.create_session(session_id, metadata)
        logger.info(f"Created session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Get session information.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session dict or None if not found
        """
        return self.db.get_session(session_id)
    
    def add_user_message(self, session_id: str, message: str):
        """
        Add a user message to the session.
        
        Args:
            session_id: Session identifier
            message: User message content
        """
        self.db.add_message(session_id, "user", message)
        logger.debug(f"Added user message to session {session_id}")
    
    def add_assistant_message(self, session_id: str, message: str):
        """
        Add an assistant message to the session.
        
        Args:
            session_id: Session identifier
            message: Assistant message content
        """
        self.db.add_message(session_id, "assistant", message)
        logger.debug(f"Added assistant message to session {session_id}")
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> list[dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of message dicts with role, content, timestamp
        """
        return self.db.get_messages(session_id, limit)
    
    def add_research_turn(
        self,
        session_id: str,
        query: str,
        search_queries: list[str],
        urls_opened: list[str],
        context_snippets: list[dict],
        answer: str,
        citations: list[dict],
        confidence: str
    ):
        """
        Record a complete research turn.
        
        Args:
            session_id: Session identifier
            query: User query
            search_queries: List of search queries issued
            urls_opened: List of URLs fetched
            context_snippets: List of context snippet dicts
            answer: Generated answer
            citations: List of citation dicts
            confidence: Confidence level (high/medium/low)
        """
        # Prepare context snippets for storage
        snippet_data = [
            {
                'url': s.url,
                'title': s.title,
                'domain': s.domain,
                'snippet': s.text[:500]  # Store truncated version
            }
            for s in context_snippets
        ]
        
        # Prepare citations for storage
        citation_data = [
            {
                'title': c.title,
                'domain': c.domain,
                'url': c.url
            }
            for c in citations
        ]
        
        self.db.add_research_turn(
            session_id=session_id,
            query=query,
            search_queries=search_queries,
            urls_opened=urls_opened,
            context_snippets=snippet_data,
            answer=answer,
            citations=citation_data,
            confidence=confidence
        )
        
        logger.info(f"Recorded research turn for session {session_id}")
    
    def get_research_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> list[dict]:
        """
        Get research turn history for a session.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of turns
            
        Returns:
            List of research turn dicts
        """
        return self.db.get_research_turns(session_id, limit)
    
    def list_sessions(self, limit: int = 50) -> list[dict]:
        """
        List recent sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dicts
        """
        return self.db.list_sessions(limit)
    
    def delete_session(self, session_id: str):
        """
        Delete a session and all its data.
        
        Args:
            session_id: Session identifier
        """
        self.db.delete_session(session_id)
        logger.info(f"Deleted session: {session_id}")
    
    def get_context_for_query(
        self,
        session_id: str,
        max_messages: int = 10
    ) -> list[dict]:
        """
        Get relevant context for a new query.
        
        Returns recent conversation messages in a format suitable for LLM.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum messages to include
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = self.db.get_messages(session_id, limit=max_messages)
        
        # Convert to LLM format
        return [
            {
                'role': msg['role'],
                'content': msg['content']
            }
            for msg in messages
        ]
