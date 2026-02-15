"""
Configuration management for the Deep Research Agent.
Loads environment variables and provides centralized configuration access.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management."""
    
    # Project paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    EVALUATION_DIR = BASE_DIR / "evaluation"
    
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    EVALUATION_DIR.mkdir(exist_ok=True)
    
    # LLM Configuration (Main - for orchestrator, refiner, answer gen)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    
    # Strategist LLM Configuration (Can be different/cheaper for planning)
    STRATEGIST_LLM_PROVIDER = os.getenv("STRATEGIST_LLM_PROVIDER", "openai")
    STRATEGIST_LLM_MODEL = os.getenv("STRATEGIST_LLM_MODEL", "gpt-3.5-turbo")
    STRATEGIST_LLM_TEMPERATURE = float(os.getenv("STRATEGIST_LLM_TEMPERATURE", "0.3"))
    
    # Refiner LLM Configuration (Can use cheaper/faster model for refinement)
    REFINER_LLM_PROVIDER = os.getenv("REFINER_LLM_PROVIDER", LLM_PROVIDER)
    REFINER_LLM_MODEL = os.getenv("REFINER_LLM_MODEL", LLM_MODEL)
    REFINER_LLM_TEMPERATURE = float(os.getenv("REFINER_LLM_TEMPERATURE", "0.2"))
    
    # ContextResolver LLM Configuration (For query refinement)
    CONTEXT_RESOLVER_LLM_PROVIDER = os.getenv("CONTEXT_RESOLVER_LLM_PROVIDER", LLM_PROVIDER)
    CONTEXT_RESOLVER_LLM_MODEL = os.getenv("CONTEXT_RESOLVER_LLM_MODEL", LLM_MODEL)
    CONTEXT_RESOLVER_LLM_TEMPERATURE = float(os.getenv("CONTEXT_RESOLVER_LLM_TEMPERATURE", "0.4"))
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    
    # Search Configuration
    SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily")
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    MAX_PAGES_TO_FETCH = int(os.getenv("MAX_PAGES_TO_FETCH", "5"))
    
    # Context Configuration
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
    
    # Agent Configuration
    ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    
    # Database Configuration
    DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATA_DIR / "sessions.db"))
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration and return list of missing items."""
        missing = []
        
        # Check LLM provider API key
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        elif cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY")
        elif cls.LLM_PROVIDER == "google" and not cls.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        elif cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        
        # Check search provider API key
        if cls.SEARCH_PROVIDER == "tavily" and not cls.TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY")
        elif cls.SEARCH_PROVIDER == "serper" and not cls.SERPER_API_KEY:
            missing.append("SERPER_API_KEY")
        
        return missing
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration dictionary."""
        return {
            "provider": cls.LLM_PROVIDER,
            "model": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "api_key": getattr(cls, f"{cls.LLM_PROVIDER.upper()}_API_KEY"),
        }
    
    @classmethod
    def get_strategist_llm_config(cls) -> dict:
        """Get Strategist-specific LLM configuration."""
        return {
            "provider": cls.STRATEGIST_LLM_PROVIDER,
            "model": cls.STRATEGIST_LLM_MODEL,
            "temperature": cls.STRATEGIST_LLM_TEMPERATURE,
            "api_key": getattr(cls, f"{cls.STRATEGIST_LLM_PROVIDER.upper()}_API_KEY"),
        }
    
    @classmethod
    def get_refiner_llm_config(cls) -> dict:
        """Get Refiner-specific LLM configuration."""
        return {
            "provider": cls.REFINER_LLM_PROVIDER,
            "model": cls.REFINER_LLM_MODEL,
            "temperature": cls.REFINER_LLM_TEMPERATURE,
            "api_key": getattr(cls, f"{cls.REFINER_LLM_PROVIDER.upper()}_API_KEY"),
        }
    
    @classmethod
    def get_context_resolver_llm_config(cls) -> dict:
        """Get ContextResolver-specific LLM configuration."""
        return {
            "provider": cls.CONTEXT_RESOLVER_LLM_PROVIDER,
            "model": cls.CONTEXT_RESOLVER_LLM_MODEL,
            "temperature": cls.CONTEXT_RESOLVER_LLM_TEMPERATURE,
            "api_key": getattr(cls, f"{cls.CONTEXT_RESOLVER_LLM_PROVIDER.upper()}_API_KEY"),
        }
    
    @classmethod
    def get_search_config(cls) -> dict:
        """Get search configuration dictionary."""
        return {
            "provider": cls.SEARCH_PROVIDER,
            "api_key": cls.TAVILY_API_KEY if cls.SEARCH_PROVIDER == "tavily" else cls.SERPER_API_KEY,
            "max_results": cls.MAX_SEARCH_RESULTS,
        }

# Singleton instance
config = Config()
