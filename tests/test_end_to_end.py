"""End-to-End integration tests for Moment pipeline."""

import json
from unittest.mock import MagicMock, patch
from ai.whisper.transcriber import transcribe_audio
from ai.llm.processor import process_transcript

def test_end_to_end_pipeline(silent_wav, test_config, mock_groq_whisper_response, mock_groq_llm_response, monkeypatch):
    """Test the full pipeline from audio file to MeetingIntelligence."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")

    # 1. Mock Groq Whisper HTTP Request
    mock_whisper_resp = MagicMock()
    mock_whisper_resp.status_code = 200
    mock_whisper_resp.json.return_value = mock_groq_whisper_response

    # 2. Mock Groq LLM SDK Client
    mock_llm_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = mock_groq_llm_response
    mock_llm_client.chat.completions.create.return_value = mock_completion

    with patch("requests.post", return_value=mock_whisper_resp):
        with patch("ai.llm.client.Groq", return_value=mock_llm_client):
            
            # Step A: Transcribe Audio
            result = transcribe_audio(silent_wav, "m1", config=test_config)
            assert result["status"] == "completed"
            
            # Extract transcript text
            transcript_text = "\n".join(seg["text"] for seg in result["segments"])
            assert "launch tomorrow" in transcript_text.lower()
            
            # Step B: Process with LLM
            intelligence = process_transcript(transcript_text)
            
            # Step C: Validate Intelligence
            assert intelligence.summary == "Discussed the upcoming launch."
            assert intelligence.decisions[0].decision == "Launch tomorrow"
            assert intelligence.action_items[0].owner == "Alice"
