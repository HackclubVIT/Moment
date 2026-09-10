"""Speech-to-text interface and engine selection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Settings
from ..models import Transcript


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path) -> Transcript:
        """Convert an audio file into a timestamped transcript."""


def build_transcriber(settings: Settings) -> Transcriber:
    """Resolve the configured STT engine.

    - 'api'   -> OpenAI-compatible /audio/transcriptions (Whisper hosted)
    - 'local' -> faster-whisper running on this machine (offline)
    - 'auto'  -> api if an OpenAI key is present, else local
    """
    from .openai_stt import OpenAISTT
    from .whisper_local import WhisperLocal

    engine = settings.stt_engine
    if engine == "api":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for stt_engine=api")
        return OpenAISTT(settings)
    if engine == "local":
        return WhisperLocal(model_size=settings.local_stt_model)
    if settings.openai_api_key:
        return OpenAISTT(settings)
    return WhisperLocal(model_size=settings.local_stt_model)
