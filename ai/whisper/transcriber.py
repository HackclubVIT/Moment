"""Transcription orchestrator — the single public entry point.

This module provides ``transcribe_audio()``, which is the only function
that Raghavraj's backend (or anyone else) needs to call.  It handles:

  1. Audio validation
  2. Long-audio chunking (>2 hr → overlapping segments)
  3. Groq cloud → local fallback routing
  4. Transcript cleaning
  5. Saving results to disk
"""

from __future__ import annotations

import base64
import json
import logging
import math
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from .cleaner import clean_transcript
from .config import WhisperConfig, load_config

logger = logging.getLogger(__name__)

# Type alias for the progress callback
ProgressCallback = Callable[[str, float], None] | None


def transcribe_audio(
    audio_path: str,
    meeting_id: str,
    language: str = "auto",
    translate: bool = True,
    on_progress: ProgressCallback = None,
    config: WhisperConfig | None = None,
) -> dict:
    """Transcribe an audio file with speaker diarization.

    This function **never raises**.  On failure it returns an error-response
    dict with ``"status": "error"`` so the backend always gets predictable
    data.

    Args:
        audio_path: Path to the WAV audio file.
        meeting_id: Unique meeting identifier.
        language: Language code or ``"auto"`` for auto-detection.
        translate: If True, include English translation for non-English audio.
        on_progress: Optional ``callback(stage, progress_0_to_1)``.
        config: Optional config override (uses ``.env`` by default).

    Returns:
        Transcript result dict matching the project JSON schema.
    """
    cfg = config or load_config()
    t0 = time.time()

    def _emit(stage: str, progress: float = 0.0) -> None:
        if on_progress:
            on_progress(stage, progress)

    try:
        # ── 1. Validate ────────────────────────────────────────────────
        _emit("validating", 0.0)
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            return _error_response(meeting_id, f"File not found: {audio_path}")
        if not audio_path_obj.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            return _error_response(meeting_id, f"Unsupported format: {audio_path_obj.suffix}")

        try:
            info = sf.info(str(audio_path))
        except Exception as exc:
            return _error_response(meeting_id, f"Cannot read audio file: {exc}")

        duration_hrs = info.duration / 3600

        # ── 2. Chunk if needed ─────────────────────────────────────────
        _emit("chunking", 0.05)
        chunk_paths = _maybe_chunk(audio_path, duration_hrs, cfg)

        # ── 3. Transcribe (Groq → local fallback) ──────────────────────
        all_segments: list[dict] = []
        inference_mode = "groq"
        model_used = cfg.model_size
        detected_language = language

        for idx, chunk_path in enumerate(chunk_paths):
            chunk_progress = 0.1 + 0.7 * (idx / max(len(chunk_paths), 1))
            _emit("transcribing", chunk_progress)

            result = _transcribe_chunk(chunk_path, language, translate, cfg)

            if result is None:
                return _error_response(
                    meeting_id,
                    "Transcription failed on both Groq and local fallback",
                )

            inference_mode = result["metadata"]["inference_mode"]
            model_used = result["metadata"]["model"]
            detected_language = result["metadata"].get("language_detected", language)

            # Offset timestamps for chunks beyond the first
            if idx > 0:
                chunk_offset = _chunk_offset(idx, info.duration, cfg)
                for seg in result["segments"]:
                    seg["start"] += chunk_offset
                    seg["end"] += chunk_offset
                    for w in seg.get("words", []):
                        w["start"] += chunk_offset
                        w["end"] += chunk_offset

            all_segments.extend(result["segments"])

        # ── 4. Clean ───────────────────────────────────────────────────
        _emit("cleaning", 0.85)
        all_segments = clean_transcript(all_segments)

        # ── 5. Build final result ──────────────────────────────────────
        # Resolve target transcript path
        out_dir = cfg.transcripts_path()
        target_path = str((out_dir / f"{meeting_id}.json").resolve())

        transcript_result = {
            "meeting_id": meeting_id,
            "status": "completed",
            "error": None,
            "metadata": {
                "audio_duration_seconds": round(info.duration, 2),
                "processing_time_seconds": round(time.time() - t0, 2),
                "inference_mode": inference_mode,
                "model": model_used,
                "language_detected": detected_language,
                "num_speakers": len({s.get("speaker", "SPEAKER_00") for s in all_segments}),
                "num_segments": len(all_segments),
                "audio_path": str(audio_path),
                "transcript_path": target_path,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "segments": all_segments,
        }

        # Save to disk
        _save_transcript(transcript_result, meeting_id, cfg)

        _emit("completed", 1.0)
        logger.info(
            "Transcription complete: %s (%.1fs, %d segments)",
            meeting_id,
            time.time() - t0,
            len(all_segments),
        )
        return transcript_result

    except Exception as exc:
        logger.exception("Transcription failed for %s", meeting_id)
        _emit("error", 0.0)
        return _error_response(meeting_id, str(exc))


# ── internal helpers ────────────────────────────────────────────────────


def _transcribe_chunk(
    chunk_path: str,
    language: str,
    translate: bool,
    cfg: WhisperConfig,
) -> dict | None:
    """Try Groq first, fall back to local. Returns None on total failure."""

    # Try Groq
    if cfg.groq_api_key:
        result = _try_groq(chunk_path, language, translate, cfg)
        if result is not None:
            return result
        logger.warning("Groq failed — falling back to local inference")

    # Local fallback
    return _try_local(chunk_path, language, translate, cfg)


def _try_groq(
    audio_path: str,
    language: str,
    translate: bool,
    cfg: WhisperConfig,
) -> dict | None:
    """Send audio to Groq Whisper API. Returns None on failure."""
    try:
        from .groq_whisper import GroqWhisper

        client = GroqWhisper(config=cfg)
        return client.transcribe(audio_path, language=language, translate=translate)
    except Exception as exc:
        logger.warning("Groq request failed: %s", exc)
        return None


def _try_local(
    audio_path: str,
    language: str,
    translate: bool,
    cfg: WhisperConfig,
) -> dict | None:
    """Run WhisperX locally.  Returns None on failure."""
    try:
        from .local_whisper import LocalWhisperX

        engine = LocalWhisperX(config=cfg)
        return engine.transcribe(audio_path, language=language, translate=translate)
    except Exception as exc:
        logger.exception("Local transcription failed: %s", exc)
        return None


def _maybe_chunk(
    audio_path: str,
    duration_hrs: float,
    cfg: WhisperConfig,
) -> list[str]:
    """Split audio into overlapping chunks if it exceeds ``max_chunk_hours``.

    Returns a list of file paths (original file if no chunking needed).
    """
    if duration_hrs <= cfg.max_chunk_hours:
        return [audio_path]

    data, sr = sf.read(audio_path, dtype="float32")
    chunk_samples = int(cfg.max_chunk_hours * 3600 * sr)
    overlap_samples = int(30 * sr)  # 30-second overlap

    # If chunk is smaller than overlap, no point in chunking
    if chunk_samples <= overlap_samples:
        return [audio_path]

    chunks: list[str] = []
    start = 0
    while start < len(data):
        end = min(start + chunk_samples, len(data))
        chunk_data = data[start:end]

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, chunk_data, sr, subtype="PCM_16")
        chunks.append(tmp.name)

        if end >= len(data):
            break
        start = end - overlap_samples  # overlap for continuity

    logger.info("Split %.1f-hour audio into %d chunks", duration_hrs, len(chunks))
    return chunks


def _chunk_offset(chunk_index: int, total_duration: float, cfg: WhisperConfig) -> float:
    """Calculate the timestamp offset for a given chunk index."""
    chunk_seconds = cfg.max_chunk_hours * 3600
    overlap = 30.0
    return chunk_index * (chunk_seconds - overlap)


def _save_transcript(result: dict, meeting_id: str, cfg: WhisperConfig) -> str:
    """Write transcript JSON to the transcripts directory."""
    out_dir = cfg.transcripts_path()
    out_dir.mkdir(parents=True, exist_ok=True)

    filepath = out_dir / f"{meeting_id}.json"
    filepath.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return str(filepath.resolve())


def _error_response(meeting_id: str, error_msg: str) -> dict:
    """Build a structured error response matching the output schema."""
    return {
        "meeting_id": meeting_id,
        "status": "error",
        "error": error_msg,
        "metadata": {
            "audio_duration_seconds": 0,
            "processing_time_seconds": 0,
            "inference_mode": "none",
            "model": "",
            "language_detected": "",
            "num_speakers": 0,
            "num_segments": 0,
            "audio_path": "",
            "transcript_path": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "segments": [],
    }
