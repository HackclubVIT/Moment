"""Centralized configuration for the WhisperX module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from ai/whisper/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class WhisperConfig:
    """Immutable configuration for the WhisperX pipeline.

    All token/secret fields default to empty strings so the app
    can start without them (e.g. for recording-only mode).
    Actual values are expected in the .env file.
    """

    # Groq Cloud
    groq_api_key: str = ""
    groq_model: str = "whisper-large-v3"

    # HuggingFace (required for local pyannote diarization)
    hf_token: str = ""

    # Whisper configuration
    model_size: str = "whisper-large-v3"
    language: str = "auto"
    translate: bool = True
    max_chunk_hours: float = 2.0

    # Audio
    sample_rate: int = 16000
    channels: int = 1

    # Storage (relative to project root)
    recordings_dir: str = "data/recordings"
    transcripts_dir: str = "data/transcripts"

    # Local fallback
    local_model_size: str = "large-v3"
    local_compute_type: str = "auto"

    def recordings_path(self) -> Path:
        """Absolute path to the recordings directory."""
        return _PROJECT_ROOT / self.recordings_dir

    def transcripts_path(self) -> Path:
        """Absolute path to the transcripts directory."""
        return _PROJECT_ROOT / self.transcripts_dir


def load_config() -> WhisperConfig:
    """Build a WhisperConfig from environment variables.

    Missing env vars fall back to dataclass defaults.
    """
    return WhisperConfig(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3"),
        hf_token=os.getenv("HF_TOKEN", ""),
        model_size=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
        language=os.getenv("WHISPER_LANGUAGE", "auto"),
        translate=os.getenv("WHISPER_TRANSLATE", "true").lower() == "true",
        max_chunk_hours=float(os.getenv("WHISPER_MAX_CHUNK_HOURS", "2")),
        sample_rate=int(os.getenv("WHISPER_SAMPLE_RATE", "16000")),
        channels=int(os.getenv("WHISPER_CHANNELS", "1")),
        recordings_dir=os.getenv("WHISPER_RECORDINGS_DIR", "data/recordings"),
        transcripts_dir=os.getenv("WHISPER_TRANSCRIPTS_DIR", "data/transcripts"),
        local_model_size=os.getenv("WHISPER_LOCAL_MODEL", "large-v3"),
        local_compute_type=os.getenv("WHISPER_LOCAL_COMPUTE", "auto"),
    )
