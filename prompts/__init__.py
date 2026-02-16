"""Prompts for the Deep Research Agent."""

from .research_strategy_prompts import (
    RESEARCH_STRATEGY_SYSTEM_PROMPT,
    build_user_prompt as build_strategy_user_prompt
)
from .refiner_prompts import REFINER_SYSTEM_PROMPT
from .context_resolver_prompts import (
    CONTEXT_RESOLVER_SYSTEM_PROMPT,
    build_user_prompt as build_context_resolver_user_prompt
)
from .answer_generator_prompts import (
    ANSWER_GENERATOR_SYSTEM_PROMPT,
    build_user_prompt as build_answer_generator_user_prompt
)
from .llm_judge_prompts import (
    LLM_JUDGE_SYSTEM_PROMPT,
    build_evaluation_prompt
)

__all__ = [
    "RESEARCH_STRATEGY_SYSTEM_PROMPT",
    "build_strategy_user_prompt",
    "REFINER_SYSTEM_PROMPT",
    "CONTEXT_RESOLVER_SYSTEM_PROMPT",
    "build_context_resolver_user_prompt",
    "ANSWER_GENERATOR_SYSTEM_PROMPT",
    "build_answer_generator_user_prompt",
    "LLM_JUDGE_SYSTEM_PROMPT",
    "build_evaluation_prompt",
]
