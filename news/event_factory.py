from __future__ import annotations

import os

from .event_extractor import LLMEventExtractor, RuleBasedEventExtractor
from .llm_client import create_llm_prompt_fn


def create_event_extractor(use_llm: bool = False) -> LLMEventExtractor | RuleBasedEventExtractor:
    if not use_llm:
        return RuleBasedEventExtractor()

    client = create_llm_prompt_fn(
        api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        model=os.environ.get("LLM_MODEL"),
    )
    if client is None:
        print("LLM disabled: set OPENAI_API_KEY or LLM_API_KEY to enable.")
        return RuleBasedEventExtractor()
    return LLMEventExtractor(client=client)
