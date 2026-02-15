"""
Text processing utilities for extracting and cleaning web content.
"""
import re
import logging
from typing import Optional
from urllib.parse import urlparse
import html2text
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract and clean text from HTML content."""
    
    def __init__(self):
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = True
        self.html2text.ignore_emphasis = False
        
    def extract_from_html(self, html_content: str, url: str = "") -> dict:
        """
        Extract clean text from HTML content.
        
        Returns:
            dict with keys: text, title, word_count
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string.strip() if soup.title.string else ""
            elif soup.find('h1'):
                title = soup.find('h1').get_text().strip()
            
            # Try trafilatura first (best quality)
            try:
                import trafilatura
                text = trafilatura.extract(html_content, include_comments=False, include_tables=True)
                if text and len(text) > 100:
                    return {
                        "text": text,
                        "title": title,
                        "word_count": len(text.split())
                    }
            except Exception as e:
                logger.debug(f"Trafilatura extraction failed: {e}")
            
            # Fallback to BeautifulSoup
            # Get main content (try common containers first)
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_=re.compile('content|main|article', re.I)) or
                soup.find('body')
            )
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text
            text = self._clean_text(text)
            
            return {
                "text": text,
                "title": title,
                "word_count": len(text.split())
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return {
                "text": "",
                "title": "",
                "word_count": 0
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove very short lines (likely navigation/footer)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 3 or line.strip() == '']
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    def truncate_text(self, text: str, max_tokens: int, tokenizer_fn=None) -> str:
        """
        Truncate text to approximate token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum token count
            tokenizer_fn: Optional function to count tokens accurately
        """
        if tokenizer_fn:
            if tokenizer_fn(text) <= max_tokens:
                return text
            
            # Binary search to find the right length
            words = text.split()
            left, right = 0, len(words)
            
            while left < right:
                mid = (left + right + 1) // 2
                candidate = ' '.join(words[:mid])
                if tokenizer_fn(candidate) <= max_tokens:
                    left = mid
                else:
                    right = mid - 1
            
            return ' '.join(words[:left])
        else:
            # Rough estimate: 1 token ≈ 4 characters
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars]
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url
    
    def create_snippet(self, text: str, max_words: int = 100) -> str:
        """Create a snippet from text."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words]) + '...'


# Singleton instance
text_extractor = TextExtractor()
