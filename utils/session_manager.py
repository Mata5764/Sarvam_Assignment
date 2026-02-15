"""
Session Manager - Handles persistent session storage and timeout management.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)


@dataclass
class TurnData:
    """Data for a single research turn."""
    turn_id: int
    query: str
    strategy: str
    search_queries: List[str]
    urls_opened: List[str]
    refined_data: List[dict]
    raw_search_results: List[dict]  # NEW: Store raw content from searches
    final_answer: str
    citations: List[dict]
    timestamp: str
    duration_ms: int


class SessionManager:
    """
    Manages session persistence with timeout-based cleanup.
    
    Sessions are stored in: data/sessions/{session_id}/
        - conversation.json: Chat history
        - turns.json: Detailed turn-by-turn data
        - metadata.json: Session info (last_activity, created_at)
    """
    
    def __init__(self, base_dir: str = "data/sessions", timeout_minutes: int = 30):
        self.base_dir = Path(base_dir)
        self.timeout_minutes = timeout_minutes
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SessionManager initialized with {timeout_minutes}min timeout")
    
    def create_session(self) -> str:
        """Create a new session with unique ID."""
        session_id = str(uuid.uuid4())
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Initialize metadata
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "status": "active"
        }
        
        self._save_json(session_dir / "metadata.json", metadata)
        self._save_json(session_dir / "conversation.json", [])
        self._save_json(session_dir / "turns.json", [])
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if session is active (not timed out)."""
        session_dir = self.base_dir / session_id
        
        if not session_dir.exists():
            return False
        
        metadata_file = session_dir / "metadata.json"
        if not metadata_file.exists():
            return False
        
        metadata = self._load_json(metadata_file)
        
        # Check if manually ended
        if metadata.get("status") == "ended":
            return False
        
        # Check timeout
        last_activity = datetime.fromisoformat(metadata["last_activity"])
        timeout = timedelta(minutes=self.timeout_minutes)
        
        if datetime.now() - last_activity > timeout:
            # Session timed out - mark as ended
            self._end_session_internal(session_id, "timeout")
            return False
        
        return True
    
    def update_last_activity(self, session_id: str):
        """Update session's last activity timestamp."""
        session_dir = self.base_dir / session_id
        metadata_file = session_dir / "metadata.json"
        
        if metadata_file.exists():
            metadata = self._load_json(metadata_file)
            metadata["last_activity"] = datetime.now().isoformat()
            self._save_json(metadata_file, metadata)
    
    def save_conversation_message(
        self,
        session_id: str,
        role: str,
        content: str
    ):
        """Save a conversation message (user or assistant)."""
        if not self.is_session_active(session_id):
            logger.warning(f"Cannot save message - session {session_id} inactive")
            return
        
        session_dir = self.base_dir / session_id
        conv_file = session_dir / "conversation.json"
        
        conversation = self._load_json(conv_file)
        conversation.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        self._save_json(conv_file, conversation)
        self.update_last_activity(session_id)
        logger.debug(f"Saved {role} message to session {session_id}")
    
    def save_turn(
        self,
        session_id: str,
        turn_data: TurnData
    ):
        """Save detailed turn data."""
        if not self.is_session_active(session_id):
            logger.warning(f"Cannot save turn - session {session_id} inactive")
            return
        
        session_dir = self.base_dir / session_id
        turns_file = session_dir / "turns.json"
        
        turns = self._load_json(turns_file)
        turns.append(asdict(turn_data))
        
        self._save_json(turns_file, turns)
        self.update_last_activity(session_id)
        logger.debug(f"Saved turn {turn_data.turn_id} to session {session_id}")
    
    def load_conversation_history(self, session_id: str) -> List[dict]:
        """Load conversation history for a session."""
        if not self.is_session_active(session_id):
            logger.warning(f"Session {session_id} inactive, returning empty history")
            return []
        
        session_dir = self.base_dir / session_id
        conv_file = session_dir / "conversation.json"
        
        if not conv_file.exists():
            return []
        
        self.update_last_activity(session_id)
        return self._load_json(conv_file)
    
    def get_turn_history(self, session_id: str) -> List[dict]:
        """Get all turn history for a session."""
        if not self.is_session_active(session_id):
            return []
        
        session_dir = self.base_dir / session_id
        turns_file = session_dir / "turns.json"
        
        if not turns_file.exists():
            return []
        
        return self._load_json(turns_file)
    
    def get_turn_count(self, session_id: str) -> int:
        """Get number of turns in a session."""
        turns = self.get_turn_history(session_id)
        return len(turns)
    
    def end_session(self, session_id: str, reason: str = "manual"):
        """Explicitly end a session."""
        self._end_session_internal(session_id, reason)
        logger.info(f"Session {session_id} ended: {reason}")
    
    def _end_session_internal(self, session_id: str, reason: str):
        """Internal method to end session."""
        session_dir = self.base_dir / session_id
        metadata_file = session_dir / "metadata.json"
        
        if metadata_file.exists():
            metadata = self._load_json(metadata_file)
            metadata["status"] = "ended"
            metadata["ended_at"] = datetime.now().isoformat()
            metadata["end_reason"] = reason
            self._save_json(metadata_file, metadata)
    
    def cleanup_old_sessions(self, days: int = 7):
        """Delete sessions older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0
        
        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                metadata = self._load_json(metadata_file)
                created_at = datetime.fromisoformat(metadata["created_at"])
                
                if created_at < cutoff:
                    import shutil
                    shutil.rmtree(session_dir)
                    deleted += 1
                    logger.info(f"Deleted old session: {session_dir.name}")
        
        logger.info(f"Cleaned up {deleted} old sessions")
        return deleted
    
    def _load_json(self, file_path: Path) -> dict:
        """Load JSON from file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {} if "metadata" in str(file_path) else []
    
    def _save_json(self, file_path: Path, data):
        """Save JSON to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
