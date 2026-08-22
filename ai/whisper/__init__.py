"""Moment AI — WhisperX audio capture & transcription module.

Public API::

    from ai.whisper import transcribe_audio, AudioRecorder

    # Record
    rec = AudioRecorder()
    rec.start()
    path = rec.stop()

    # Transcribe
    result = transcribe_audio(audio_path=path, meeting_id="meeting_123")
"""

from .config import WhisperConfig, load_config
from .groq_whisper import GroqWhisper
from .recorder import AudioRecorder
from .transcriber import transcribe_audio

__all__ = [
    "transcribe_audio",
    "AudioRecorder",
    "GroqWhisper",
    "WhisperConfig",
    "load_config",
]
