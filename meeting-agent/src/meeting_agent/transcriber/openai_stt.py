"""OpenAI-compatible speech-to-text (Whisper hosted, e.g. api.openai.com or any compatible provider)."""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..models import Segment, Transcript
from .base import Transcriber


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class OpenAISTT(Transcriber):
    def __init__(self, settings: Settings):
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        self.model = settings.openai_stt_model

    def transcribe(self, audio_path) -> Transcript:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        with open(audio_path, "rb") as fh:
            resp = client.audio.transcriptions.create(
                model=self.model,
                file=fh,
                response_format="verbose_json",
            )

        segments: list[Segment] = []
        for item in _get(resp, "segments", []) or []:
            text = (_get(item, "text") or "").strip()
            if not text:
                continue
            segments.append(
                Segment(
                    start=float(_get(item, "start", 0.0) or 0.0),
                    end=float(_get(item, "end", 0.0) or 0.0),
                    text=text,
                )
            )
        if not segments:
            full = (_get(resp, "text") or "").strip()
            if full:
                segments.append(Segment(text=full))

        return Transcript(path=str(Path(audio_path)), language=_get(resp, "language"), segments=segments)
