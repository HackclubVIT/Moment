"""Tests for the full Whisper workflow."""

from unittest.mock import MagicMock, patch
import pytest

from ai.whisper.transcriber import transcribe_audio
from ai.whisper.config import WhisperConfig

def test_missing_audio_file(test_config):
    """Test validation: missing file returns error."""
    result = transcribe_audio("/fake/path.wav", "m1", config=test_config)
    assert result["status"] == "error"
    assert "not found" in result["error"].lower()

def test_corrupt_audio_file(corrupt_wav, test_config):
    """Test validation: corrupt file returns error."""
    result = transcribe_audio(corrupt_wav, "m1", config=test_config)
    assert result["status"] == "error"
    assert "cannot read" in result["error"].lower()

def test_transcribe_audio_groq_success(silent_wav, test_config, mock_groq_whisper_response):
    """Test successful transcription using Groq cloud."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_groq_whisper_response

    with patch("requests.post", return_value=mock_resp):
        result = transcribe_audio(silent_wav, "m1", config=test_config)
        
    assert result["status"] == "completed"
    assert result["metadata"]["inference_mode"] == "groq"
    assert result["segments"][0]["text"] == "Hello, this is a test meeting. We decided to launch tomorrow."

def test_transcribe_audio_local_fallback(silent_wav, test_config):
    """Test fallback to local WhisperX when Groq fails."""
    # Force Groq to fail
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    
    # Mock local WhisperX
    mock_local = MagicMock()
    mock_local.transcribe.return_value = {
        "status": "completed",
        "error": None,
        "metadata": {
            "inference_mode": "local",
            "model": "base",
            "language_detected": "en",
            "num_speakers": 1,
            "num_segments": 1,
        },
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Local text."}]
    }

    with patch("requests.post", return_value=mock_resp):
        with patch("ai.whisper.transcriber._try_local", return_value=mock_local.transcribe.return_value):
            result = transcribe_audio(silent_wav, "m1", config=test_config)
            
    assert result["status"] == "completed"
    assert result["metadata"]["inference_mode"] == "local"
    assert result["segments"][0]["text"] == "Local text."
