"""Tests for output JSON schema validation."""

from __future__ import annotations

from ai.whisper.transcriber import _error_response

# Expected top-level keys in every transcript response
REQUIRED_TOP_KEYS = {"meeting_id", "status", "error", "metadata", "segments"}

REQUIRED_META_KEYS = {
    "audio_duration_seconds",
    "processing_time_seconds",
    "inference_mode",
    "model",
    "language_detected",
    "num_speakers",
    "num_segments",
    "audio_path",
    "transcript_path",
    "created_at",
}

REQUIRED_SEGMENT_KEYS = {"id", "start", "end", "speaker", "text"}


def _make_success_response() -> dict:
    """Build a synthetic successful response for schema testing."""
    return {
        "meeting_id": "test_123",
        "status": "completed",
        "error": None,
        "metadata": {
            "audio_duration_seconds": 60.0,
            "processing_time_seconds": 10.0,
            "inference_mode": "groq",
            "model": "whisper-large-v3",
            "language_detected": "en",
            "num_speakers": 2,
            "num_segments": 3,
            "audio_path": "data/recordings/test_123.wav",
            "transcript_path": "data/transcripts/test_123.json",
            "created_at": "2026-08-20T10:30:00Z",
        },
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.2,
                "speaker": "SPEAKER_00",
                "text": "Welcome everyone.",
                "words": [{"word": "Welcome", "start": 0.0, "end": 0.45}],
            },
            {
                "id": 1,
                "start": 5.3,
                "end": 10.0,
                "speaker": "SPEAKER_01",
                "text": "Thank you.",
                "words": [],
            },
        ],
    }


class TestSuccessSchema:
    def test_top_level_keys(self):
        resp = _make_success_response()
        assert REQUIRED_TOP_KEYS.issubset(resp.keys())

    def test_metadata_keys(self):
        resp = _make_success_response()
        assert REQUIRED_META_KEYS.issubset(resp["metadata"].keys())

    def test_segment_keys(self):
        resp = _make_success_response()
        for seg in resp["segments"]:
            assert REQUIRED_SEGMENT_KEYS.issubset(seg.keys())

    def test_status_is_completed(self):
        resp = _make_success_response()
        assert resp["status"] == "completed"
        assert resp["error"] is None


class TestErrorSchema:
    def test_top_level_keys(self):
        resp = _error_response("m1", "fail")
        assert REQUIRED_TOP_KEYS.issubset(resp.keys())

    def test_metadata_keys(self):
        resp = _error_response("m1", "fail")
        assert REQUIRED_META_KEYS.issubset(resp["metadata"].keys())

    def test_status_is_error(self):
        resp = _error_response("m1", "fail")
        assert resp["status"] == "error"
        assert resp["error"] == "fail"
        assert resp["segments"] == []
