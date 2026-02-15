"""
Web search interface supporting multiple search providers (Tavily, Serper).
"""
import logging
import aiohttp
from typing import Optional

from models.search_models import SearchResult

logger = logging.getLogger(__name__)


class SearchProvider:
    """Base class for search providers."""
    
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Execute search and return results."""
        raise NotImplementedError


class TavilySearch(SearchProvider):
    """Tavily search provider (recommended - 1000 free searches/month)."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.tavily.com/search"
        
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Execute search using Tavily API (async)."""
        try:
            logger.info(f"Executing Tavily search: {query}")
            
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",  # "basic" or "advanced" - more comprehensive
                "include_raw_content": True,  # Get full page content
                "include_answer": True,  # Get Tavily's AI-generated answer
                "topic": "general",  # "general" or "news"
                "days": None,  # Limit to recent results (e.g., 7 for last week)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            results = []
            for item in data.get('results', []):
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    snippet=item.get('content', ''),
                    content=item.get('raw_content', ''),  # Full page content from Tavily
                    relevance_score=item.get('score'),
                    published_date=item.get('published_date'),
                )
                
                # Extract domain
                from urllib.parse import urlparse
                try:
                    domain = urlparse(result.url).netloc
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    result.domain = domain
                except:
                    result.domain = result.url
                
                results.append(result)
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []


class SerperSearch(SearchProvider):
    """Serper.dev search provider (alternative to Tavily)."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://google.serper.dev/search"
        
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Execute search using Serper API (async)."""
        try:
            logger.info(f"Executing Serper search: {query}")
            
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'q': query,
                'num': max_results
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            results = []
            for item in data.get('organic', [])[:max_results]:
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    content=item.get('snippet', ''),  # Serper only provides snippet
                    published_date=item.get('date'),
                )
                
                # Extract domain
                from urllib.parse import urlparse
                try:
                    domain = urlparse(result.url).netloc
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    result.domain = domain
                except:
                    result.domain = result.url
                
                results.append(result)
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []


def create_search_provider(provider: str, api_key: str) -> SearchProvider:
    """
    Factory function to create appropriate search provider.
    
    Args:
        provider: Provider name ('tavily' or 'serper')
        api_key: API key for the provider
        
    Returns:
        SearchProvider instance
    """
    providers = {
        'tavily': TavilySearch,
        'serper': SerperSearch,
    }
    
    if provider not in providers:
        raise ValueError(f"Unsupported search provider: {provider}")
    
    return providers[provider](api_key)
