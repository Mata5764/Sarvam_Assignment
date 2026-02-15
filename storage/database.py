"""
Database interface using SQLite for session persistence.
"""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Database:
    """SQLite database interface for session management."""
    
    def __init__(self, db_path: str):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Conversation messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Research turns table (detailed research artifacts)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    search_queries TEXT,
                    urls_opened TEXT,
                    context_snippets TEXT,
                    answer TEXT,
                    citations TEXT,
                    confidence TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages (session_id, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_research_turns_session 
                ON research_turns (session_id, timestamp)
            """)
            
            conn.commit()
            logger.info("Database initialized")
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_session(self, session_id: str, metadata: Optional[dict] = None) -> bool:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
            metadata: Optional metadata dictionary
            
        Returns:
            True if created, False if already exists
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.utcnow().isoformat() + 'Z'
                
                cursor.execute("""
                    INSERT INTO sessions (session_id, created_at, last_active, metadata)
                    VALUES (?, ?, ?, ?)
                """, (session_id, now, now, json.dumps(metadata or {})))
                
                conn.commit()
                logger.info(f"Created session: {session_id}")
                return True
                
        except sqlite3.IntegrityError:
            logger.debug(f"Session already exists: {session_id}")
            return False
    
    def update_session_activity(self, session_id: str):
        """Update last activity timestamp for session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat() + 'Z'
            
            cursor.execute("""
                UPDATE sessions SET last_active = ? WHERE session_id = ?
            """, (now, session_id))
            
            conn.commit()
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session information."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'session_id': row['session_id'],
                    'created_at': row['created_at'],
                    'last_active': row['last_active'],
                    'metadata': json.loads(row['metadata'])
                }
            return None
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[str] = None
    ):
        """Add a message to the conversation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if not timestamp:
                timestamp = datetime.utcnow().isoformat() + 'Z'
            
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, timestamp))
            
            conn.commit()
            self.update_session_activity(session_id)
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> list[dict]:
        """Get conversation messages for a session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT role, content, timestamp 
                FROM messages 
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (session_id,))
            
            return [
                {
                    'role': row['role'],
                    'content': row['content'],
                    'timestamp': row['timestamp']
                }
                for row in cursor.fetchall()
            ]
    
    def add_research_turn(
        self,
        session_id: str,
        query: str,
        search_queries: list[str],
        urls_opened: list[str],
        context_snippets: list[dict],
        answer: str,
        citations: list[dict],
        confidence: str,
        timestamp: Optional[str] = None
    ):
        """Add a research turn with detailed artifacts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if not timestamp:
                timestamp = datetime.utcnow().isoformat() + 'Z'
            
            cursor.execute("""
                INSERT INTO research_turns (
                    session_id, query, timestamp, search_queries, 
                    urls_opened, context_snippets, answer, citations, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, query, timestamp,
                json.dumps(search_queries),
                json.dumps(urls_opened),
                json.dumps(context_snippets),
                answer,
                json.dumps(citations),
                confidence
            ))
            
            conn.commit()
            self.update_session_activity(session_id)
    
    def get_research_turns(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> list[dict]:
        """Get research turns for a session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM research_turns 
                WHERE session_id = ?
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (session_id,))
            
            return [
                {
                    'query': row['query'],
                    'timestamp': row['timestamp'],
                    'search_queries': json.loads(row['search_queries']),
                    'urls_opened': json.loads(row['urls_opened']),
                    'context_snippets': json.loads(row['context_snippets']),
                    'answer': row['answer'],
                    'citations': json.loads(row['citations']),
                    'confidence': row['confidence']
                }
                for row in cursor.fetchall()
            ]
    
    def list_sessions(self, limit: int = 50) -> list[dict]:
        """List recent sessions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_id, created_at, last_active, metadata
                FROM sessions
                ORDER BY last_active DESC
                LIMIT ?
            """, (limit,))
            
            return [
                {
                    'session_id': row['session_id'],
                    'created_at': row['created_at'],
                    'last_active': row['last_active'],
                    'metadata': json.loads(row['metadata'])
                }
                for row in cursor.fetchall()
            ]
    
    def delete_session(self, session_id: str):
        """Delete a session and all related data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM research_turns WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            conn.commit()
            logger.info(f"Deleted session: {session_id}")
