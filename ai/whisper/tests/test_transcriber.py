"""Tests for transcriber.py — orchestration and error handling."""

from __future__ import annotations

from ai.whisper.transcriber import transcribe_audio, _error_response, _maybe_chunk


class TestErrorResponse:
    """The error response must match the output JSON schema."""

    def test_has_required_fields(self):
        resp = _error_response("m1", "something broke")
        assert resp["status"] == "error"
        assert resp["error"] == "something broke"
        assert resp["meeting_id"] == "m1"
        assert resp["segments"] == []
        assert "metadata" in resp

    def test_metadata_has_required_keys(self):
        resp = _error_response("m1", "err")
        meta = resp["metadata"]
        for key in (
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
        ):
            assert key in meta, f"Missing metadata key: {key}"


class TestTranscribeAudioValidation:
    """transcribe_audio() should return structured errors, never raise."""

    def test_missing_file(self, test_config):
        result = transcribe_audio(
            audio_path="/nonexistent/file.wav",
            meeting_id="test_missing",
            config=test_config,
        )
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_unsupported_format(self, tmp_path, test_config):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not audio")
        result = transcribe_audio(
            audio_path=str(bad_file),
            meeting_id="test_bad_ext",
            config=test_config,
        )
        assert result["status"] == "error"
        assert "unsupported" in result["error"].lower()

    def test_corrupt_audio(self, corrupt_wav, test_config):
        result = transcribe_audio(
            audio_path=corrupt_wav,
            meeting_id="test_corrupt",
            config=test_config,
        )
        assert result["status"] == "error"

    def test_progress_callback_fires(self, silent_wav, test_config):
        stages: list[str] = []

        def on_progress(stage: str, pct: float):
            stages.append(stage)

        # Will fail at transcription (no Groq key, no local whisperx installed)
        # but should still fire validating + chunking callbacks before failing
        transcribe_audio(
            audio_path=silent_wav,
            meeting_id="test_progress",
            on_progress=on_progress,
            config=test_config,
        )
        assert "validating" in stages


class TestMaybeChunk:
    """Chunking logic for long audio."""

    def test_short_audio_not_chunked(self, silent_wav, test_config):
        chunks = _maybe_chunk(silent_wav, 0.001, test_config)  # 1-second file
        assert len(chunks) == 1
        assert chunks[0] == silent_wav

    def test_long_audio_is_chunked(self, tmp_path):
        """Chunking should create multiple chunk files when duration > max_chunk_hours."""
        from pathlib import Path

        from ai.whisper.config import WhisperConfig
        import numpy as np
        import soundfile as sf

        sr = 16000
        seconds = 70  # > 30s overlap so chunking can actually occur
        t = np.linspace(0, seconds, seconds * sr, endpoint=False)
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")
        audio_path = tmp_path / "long.wav"
        sf.write(str(audio_path), audio, sr, subtype="PCM_16")

        info = sf.info(str(audio_path))
        duration_hrs = info.duration / 3600

        cfg = WhisperConfig(max_chunk_hours=0.01)  # 36s chunks (> 30s overlap)
        chunks = _maybe_chunk(str(audio_path), duration_hrs, cfg)

        try:
            assert len(chunks) > 1
        finally:
            for p in chunks:
                if p != str(audio_path):
                    Path(p).unlink(missing_ok=True)
