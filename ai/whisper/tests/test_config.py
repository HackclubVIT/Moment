"""Tests for config.py — configuration loading."""

from __future__ import annotations

import os

from ai.whisper.config import WhisperConfig, load_config


def test_default_config():
    """WhisperConfig defaults are sensible."""
    cfg = WhisperConfig()
    assert cfg.model_size == "whisper-large-v3"
    assert cfg.groq_model == "whisper-large-v3"
    assert cfg.groq_api_key == ""
    assert cfg.sample_rate == 16000
    assert cfg.channels == 1
    assert cfg.max_chunk_hours == 2.0
    assert cfg.language == "auto"
    assert cfg.translate is True


def test_load_config_from_env(monkeypatch):
    """load_config() picks up environment variables."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_123")
    monkeypatch.setenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("WHISPER_LANGUAGE", "fr")
    monkeypatch.setenv("WHISPER_TRANSLATE", "false")
    monkeypatch.setenv("WHISPER_MAX_CHUNK_HOURS", "1.5")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    cfg = load_config()
    assert cfg.groq_api_key == "gsk_test_123"
    assert cfg.groq_model == "whisper-large-v3-turbo"
    assert cfg.model_size == "base"
    assert cfg.language == "fr"
    assert cfg.translate is False
    assert cfg.max_chunk_hours == 1.5
    assert cfg.hf_token == "hf_test_token"


def test_config_is_frozen():
    """WhisperConfig should be immutable (frozen dataclass)."""
    cfg = WhisperConfig()
    try:
        cfg.model_size = "tiny"  # type: ignore
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_recordings_path(test_config, tmp_dir):
    """recordings_path() resolves to an absolute Path."""
    path = test_config.recordings_path()
    assert path.is_absolute() or str(path).startswith(str(tmp_dir))


def test_transcripts_path(test_config, tmp_dir):
    """transcripts_path() resolves to an absolute Path."""
    path = test_config.transcripts_path()
    assert path.is_absolute() or str(path).startswith(str(tmp_dir))
