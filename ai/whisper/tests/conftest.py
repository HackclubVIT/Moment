"""Shared test fixtures for ai.whisper tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai.whisper.config import WhisperConfig


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def test_config(tmp_path) -> WhisperConfig:
    """Config that points storage to tmp dirs so tests don't pollute data/."""
    return WhisperConfig(
        recordings_dir=str(tmp_path / "recordings"),
        transcripts_dir=str(tmp_path / "transcripts"),
        groq_api_key="",  # no groq in tests unless mocked
        hf_token="",
    )


@pytest.fixture
def silent_wav(tmp_path) -> str:
    """Create a short silent WAV file (1 second, 16kHz mono)."""
    sr = 16000
    audio = np.zeros(sr, dtype="float32")
    path = tmp_path / "silent.wav"
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return str(path)


@pytest.fixture
def sine_wav(tmp_path) -> str:
    """Create a 3-second WAV with a 440Hz sine wave."""
    sr = 16000
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    path = tmp_path / "sine.wav"
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return str(path)


@pytest.fixture
def noise_wav(tmp_path) -> str:
    """Create a 2-second WAV with random noise."""
    sr = 16000
    rng = np.random.default_rng(42)
    audio = (rng.random(2 * sr) * 0.3).astype("float32")
    path = tmp_path / "noise.wav"
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return str(path)


@pytest.fixture
def corrupt_wav(tmp_path) -> str:
    """Create a file that looks like a WAV but has corrupt data."""
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt garbage data here")
    return str(path)
