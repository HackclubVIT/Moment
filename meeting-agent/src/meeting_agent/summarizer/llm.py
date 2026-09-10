"""LLM-backed summarizers (Anthropic Messages and OpenAI-compatible chat completions)."""

from __future__ import annotations

from ..models import MeetingSummary
from ..parsing import Transcript
from .base import Summarizer, extract_json
from .extractive import ExtractiveSummarizer

_SUMMARY_SYSTEM = """You are a meticulous executive assistant. Convert the provided meeting transcript into structured meeting minutes.

Respond with a single JSON object ONLY, using exactly these keys:
- "title": string
- "date": "YYYY-MM-DD"
- "attendees": array of strings (empty array if unknown — never invent names)
- "executive_summary": string, 2-4 sentences
- "key_topics": array of strings
- "discussion_points": array of {"topic": string, "summary": string}
- "decisions": array of strings
- "action_items": array of {"task": string, "owner": string|null, "due": string|null}
- "open_questions": array of strings
- "next_steps": array of strings

Only include facts grounded in the transcript. Do not invent attendees, decisions, or owners."""


def _build_prompt(transcript: Transcript, title: str | None) -> str:
    header = f"Meeting title: {title or 'Untitled Meeting'}\n\nTranscript:\n"
    text = transcript.text
    if len(text) > 60000:
        text = text[:60000] + "\n...[truncated]"
    return header + text


def _parse(text: str, title: str | None, fallback: Transcript) -> MeetingSummary:
    try:
        data = extract_json(text)
        return MeetingSummary(
            **{**data, "title": data.get("title") or title or "Untitled Meeting"}
        )
    except Exception:
        # Never fail the whole pipeline on a malformed model reply — fall back to
        # the offline extractive summarizer so the user still gets a MOM.
        return ExtractiveSummarizer().summarize(fallback, title=title)


class AnthropicSummarizer(Summarizer):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self.model = model

    def summarize(self, transcript: Transcript, title: str | None = None) -> MeetingSummary:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=0.2,
            system=_SUMMARY_SYSTEM,
            messages=[{"role": "user", "content": _build_prompt(transcript, title)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return _parse(text, title, transcript)


class OpenAISummarizer(Summarizer):
    def __init__(self, api_key: str, base_url: str | None = None, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model

    def summarize(self, transcript: Transcript, title: str | None = None) -> MeetingSummary:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": _build_prompt(transcript, title)},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return _parse(text, title, transcript)
