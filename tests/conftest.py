"""Shared test fixtures for Moment workflow tests."""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from typing import Any

from ai.whisper.config import WhisperConfig

@pytest.fixture
def tmp_dir(tmp_path) -> Path:
    """Provide a temporary directory for test outputs."""
    return tmp_path

@pytest.fixture
def test_config(tmp_path) -> WhisperConfig:
    """Config that points storage to tmp dirs."""
    return WhisperConfig(
        recordings_dir=str(tmp_path / "recordings"),
        transcripts_dir=str(tmp_path / "transcripts"),
        groq_api_key="gsk_mock_test_key",  
        hf_token="",
    )

@pytest.fixture
def silent_wav(tmp_path) -> str:
    """Create a short silent WAV file (1 second)."""
    sr = 16000
    audio = np.zeros(sr, dtype="float32")
    path = tmp_path / "silent.wav"
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return str(path)

@pytest.fixture
def corrupt_wav(tmp_path) -> str:
    """Create a file with corrupt WAV data."""
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt garbage data here")
    return str(path)

@pytest.fixture
def mock_groq_whisper_response() -> dict[str, Any]:
    """Synthetic successful response from Groq Whisper."""
    return {
        "task": "transcribe",
        "language": "english",
        "duration": 5.0,
        "text": "Hello, this is a test meeting. We decided to launch tomorrow.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": "Hello, this is a test meeting. We decided to launch tomorrow.",
                "words": [
                    {"word": "Hello,", "start": 0.0, "end": 1.0},
                    {"word": "this", "start": 1.0, "end": 2.0},
                ],
            }
        ],
    }

@pytest.fixture
def mock_groq_llm_response() -> str:
    """Synthetic JSON response from Groq LLM."""
    return '''
    {
        "summary": "Discussed the upcoming launch.",
        "key_points": ["Launch is tomorrow"],
        "decisions": [
            {"decision": "Launch tomorrow", "context": "Ready to go"}
        ],
        "action_items": [
            {"task": "Prepare servers", "owner": "Alice", "deadline": "Today"}
        ]
    }
    '''
