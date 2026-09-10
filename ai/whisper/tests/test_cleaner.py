"""Tests for cleaner.py — transcript cleaning logic."""

from __future__ import annotations

from ai.whisper.cleaner import clean_transcript


def test_strips_whitespace():
    segments = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "A", "text": "  hello  "}]
    result = clean_transcript(segments)
    assert result[0]["text"] == "hello"


def test_normalizes_internal_whitespace():
    segments = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "A", "text": "hello   world"}]
    result = clean_transcript(segments)
    assert result[0]["text"] == "hello world"


def test_removes_empty_segments():
    segments = [
        {"id": 0, "start": 0.0, "end": 1.0, "speaker": "A", "text": "hello"},
        {"id": 1, "start": 1.0, "end": 2.0, "speaker": "A", "text": "   "},
        {"id": 2, "start": 2.0, "end": 3.0, "speaker": "A", "text": ""},
        {"id": 3, "start": 3.0, "end": 4.0, "speaker": "B", "text": "world"},
    ]
    result = clean_transcript(segments)
    assert len(result) == 2
    assert result[0]["text"] == "hello"
    assert result[1]["text"] == "world"


def test_removes_invalid_timestamps():
    segments = [
        {"id": 0, "start": 5.0, "end": 3.0, "speaker": "A", "text": "bad timestamps"},
        {"id": 1, "start": 2.0, "end": 2.0, "speaker": "A", "text": "zero duration"},
        {"id": 2, "start": 0.0, "end": 1.0, "speaker": "A", "text": "good"},
    ]
    result = clean_transcript(segments)
    assert len(result) == 1
    assert result[0]["text"] == "good"


def test_collapses_duplicate_consecutive():
    segments = [
        {"id": 0, "start": 0.0, "end": 1.0, "speaker": "A", "text": "hello"},
        {"id": 1, "start": 1.0, "end": 2.0, "speaker": "A", "text": "hello"},
        {"id": 2, "start": 2.0, "end": 3.0, "speaker": "B", "text": "hello"},  # different speaker
    ]
    result = clean_transcript(segments)
    assert len(result) == 2
    assert result[0]["speaker"] == "A"
    assert result[0]["end"] == 2.0  # extended
    assert result[1]["speaker"] == "B"


def test_reindexes_ids():
    segments = [
        {"id": 99, "start": 0.0, "end": 1.0, "speaker": "A", "text": "first"},
        {"id": 100, "start": 1.0, "end": 2.0, "speaker": "A", "text": "second"},
    ]
    result = clean_transcript(segments)
    assert result[0]["id"] == 0
    assert result[1]["id"] == 1


def test_does_not_mutate_input():
    segments = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "A", "text": " hi "}]
    clean_transcript(segments)
    assert segments[0]["text"] == " hi "  # original unchanged


def test_empty_input():
    assert clean_transcript([]) == []
