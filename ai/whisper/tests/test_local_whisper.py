"""Tests for local_whisper.py — local fallback engine."""

from __future__ import annotations

import pytest

from ai.whisper.local_whisper import LocalWhisperX


def test_init_does_not_load_model():
    """Model should be lazy-loaded, not on __init__."""
    engine = LocalWhisperX()
    assert engine._model is None


def test_resolve_device():
    """Device resolution should return valid device + compute type."""
    pytest.importorskip("torch")
    engine = LocalWhisperX()
    device, compute = engine._resolve_device()
    assert device in ("cuda", "cpu")
    assert compute in ("float16", "int8", "auto")
