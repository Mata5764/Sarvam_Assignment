"""LLM Client abstraction supporting multiple providers."""
from abc import ABC, abstractmethod
import logging
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai
from groq import AsyncGroq

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, messages: list[dict]) -> str:
        """Generate a response from the LLM (async)."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.3):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
    async def generate(self, messages: list[dict]) -> str:
        """Generate response using OpenAI API (async)."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except:
            return len(text) // 4  # Rough estimate


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.3):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
    async def generate(self, messages: list[dict]) -> str:
        """Generate response using Anthropic API (async)."""
        try:
            # Convert OpenAI-style messages to Anthropic format
            system_message = None
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
            
            kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": anthropic_messages,
                "temperature": self.temperature,
            }
            if system_message:
                kwargs["system"] = system_message
            
            response = await self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Rough token count estimate."""
        return len(text) // 4


class GoogleClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp", temperature: float = 0.3):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature
        
    async def generate(self, messages: list[dict]) -> str:
        """Generate response using Google Gemini API (async)."""
        try:
            # Simple conversion - just combine all messages
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n\n"
            
            # Note: Gemini SDK doesn't have native async, so we use run_in_executor
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.model.generate_content,
                prompt,
                {"temperature": self.temperature, "max_output_tokens": 4096}
            )
            return response.text
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Rough token count estimate."""
        return len(text) // 4


class GroqClient(LLMClient):
    """Groq API client (compatible with OpenAI SDK)."""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
    async def generate(self, messages: list[dict]) -> str:
        """Generate response using Groq API (async)."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Rough token count estimate."""
        return len(text) // 4


def create_llm_client(provider: str, api_key: str, model: str, temperature: float = 0.3) -> LLMClient:
    """Factory function to create appropriate LLM client."""
    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
        "groq": GroqClient,
    }
    
    if provider not in clients:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    
    return clients[provider](api_key=api_key, model=model, temperature=temperature)
