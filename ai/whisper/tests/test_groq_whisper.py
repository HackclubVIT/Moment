"""Tests for groq_whisper.py — Groq Cloud Whisper API client (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai.whisper.config import WhisperConfig
from ai.whisper.groq_whisper import GroqWhisper


def test_groq_whisper_requires_api_key(silent_wav):
    """GroqWhisper should raise ValueError if GROQ_API_KEY is not set."""
    cfg = WhisperConfig(groq_api_key="")
    client = GroqWhisper(config=cfg)
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        client.transcribe(silent_wav)


def test_groq_whisper_transcribe_mocked(silent_wav):
    """Test successful transcription and schema formatting with mocked Groq API."""
    cfg = WhisperConfig(groq_api_key="gsk_mock_key", groq_model="whisper-large-v3")
    client = GroqWhisper(config=cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "task": "transcribe",
        "language": "english",
        "duration": 1.0,
        "text": "Hello world from Groq.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Hello world from Groq.",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.3},
                    {"word": "world", "start": 0.3, "end": 0.6},
                    {"word": "from", "start": 0.6, "end": 0.8},
                    {"word": "Groq.", "start": 0.8, "end": 1.0},
                ],
            }
        ],
    }

    with patch("requests.post", return_value=mock_resp):
        result = client.transcribe(silent_wav)

    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["metadata"]["inference_mode"] == "groq"
    assert result["metadata"]["model"] == "whisper-large-v3"
    assert result["metadata"]["language_detected"] == "english"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "Hello world from Groq."
    assert len(result["segments"][0]["words"]) == 4


def test_groq_whisper_health_check_success():
    """check_health returns ok when Groq API models endpoint succeeds."""
    cfg = WhisperConfig(groq_api_key="gsk_mock_key")
    client = GroqWhisper(config=cfg)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "whisper-large-v3"},
            {"id": "whisper-large-v3-turbo"},
            {"id": "llama-3.3-70b-versatile"},
        ]
    }

    with patch("requests.get", return_value=mock_resp):
        status = client.check_health()

    assert status["status"] == "ok"
    assert "whisper-large-v3" in status["available_whisper_models"]


def test_groq_whisper_health_check_missing_key():
    """check_health returns error when no key is set."""
    cfg = WhisperConfig(groq_api_key="")
    client = GroqWhisper(config=cfg)
    status = client.check_health()
    assert status["status"] == "error"
    assert "not set" in status["message"]
