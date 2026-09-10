"""Summarizer interface, JSON helpers, and engine selection."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from ..config import Settings
from ..models import MeetingSummary
from ..parsing import Transcript


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, transcript: Transcript, title: str | None = None) -> MeetingSummary:
        """Convert a transcript into structured meeting minutes."""


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates prose / fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def build_summarizer(settings: Settings) -> Summarizer:
    """Resolve the configured summarizer engine.

    - 'anthropic'  -> AnthropicMessages LLM
    - 'openai'     -> OpenAI-compatible chat completions
    - 'offline'    -> dependency-free extractive summarizer
    - 'auto'       -> first available of anthropic / openai / offline
    """
    from .extractive import ExtractiveSummarizer
    from .llm import AnthropicSummarizer, OpenAISummarizer

    engine = settings.llm_engine
    if engine == "offline":
        return ExtractiveSummarizer()
    if engine == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for llm_engine=anthropic")
        return AnthropicSummarizer(settings.anthropic_api_key, model=settings.anthropic_model)
    if engine == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for llm_engine=openai")
        return OpenAISummarizer(
            settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_llm_model,
        )
    # auto
    if settings.anthropic_api_key:
        return AnthropicSummarizer(settings.anthropic_api_key, model=settings.anthropic_model)
    if settings.openai_api_key:
        return OpenAISummarizer(
            settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_llm_model,
        )
    return ExtractiveSummarizer()
