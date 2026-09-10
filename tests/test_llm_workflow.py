"""Tests for the full LLM workflow."""

import pytest
from unittest.mock import MagicMock, patch

from ai.llm.client import GroqLLMClient
from ai.llm.processor import process_transcript
from ai.llm.schemas import MeetingIntelligence

def test_missing_api_key_raises_error(monkeypatch):
    """Test that GroqLLMClient raises ValueError if no API key is provided."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("ai.llm.client.GROQ_API_KEY", None):
        with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
            GroqLLMClient()

def test_process_transcript_success(monkeypatch, mock_groq_llm_response):
    """Test processing a valid transcript into a MeetingIntelligence object."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = mock_groq_llm_response
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("ai.llm.client.Groq", return_value=mock_client):
        result = process_transcript("This is a transcript.")
        
    assert isinstance(result, MeetingIntelligence)
    assert result.summary == "Discussed the upcoming launch."
    assert len(result.key_points) == 1
    assert result.key_points[0] == "Launch is tomorrow"
    assert len(result.decisions) == 1
    assert result.decisions[0].decision == "Launch tomorrow"
    assert len(result.action_items) == 1
    assert result.action_items[0].task == "Prepare servers"
    assert result.action_items[0].owner == "Alice"

def test_process_transcript_empty_response(monkeypatch):
    """Test processing when LLM returns an empty content string."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = ""
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("ai.llm.client.Groq", return_value=mock_client):
        with pytest.raises(ValueError, match="empty response content"):
            process_transcript("This is a transcript.")
