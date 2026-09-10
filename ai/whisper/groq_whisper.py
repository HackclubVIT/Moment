"""Groq Cloud deployment for Whisper — lightning-fast serverless transcription.

Calls the Groq Whisper API (whisper-large-v3 or whisper-large-v3-turbo)
and formats the response into the standardized Moment transcript schema.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from .config import WhisperConfig, load_config

logger = logging.getLogger(__name__)

GROQ_AUDIO_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_AUDIO_TRANSLATIONS_URL = "https://api.groq.com/openai/v1/audio/translations"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


class GroqWhisper:
    """Inference client for Groq Speech-to-Text API."""

    def __init__(self, config: WhisperConfig | None = None) -> None:
        self._cfg = config or load_config()

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        translate: bool = True,
    ) -> dict[str, Any]:
        """Transcribe an audio file using Groq Cloud API.

        Args:
            audio_path: Path to the audio file on disk.
            language: Language code (e.g. "en", "es") or "auto".
            translate: If True and language is non-English, translate to English.

        Returns:
            Transcript result dict matching the project JSON schema.
        """
        import requests

        t0 = time.time()
        api_key = self._cfg.groq_api_key
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        audio_file_path = Path(audio_path)
        if not audio_file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Choose endpoint
        use_translation = translate and language not in ("auto", "en")
        url = GROQ_AUDIO_TRANSLATIONS_URL if use_translation else GROQ_AUDIO_TRANSCRIPTIONS_URL

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        model = self._cfg.groq_model or "whisper-large-v3"
        data: dict[str, Any] = {
            "model": model,
            "response_format": "verbose_json",
            "temperature": "0.0",
        }

        # Pass language if specified and not 'auto'
        if language and language != "auto" and not use_translation:
            data["language"] = language

        # Request word & segment timestamps if supported
        data_fields = list(data.items())
        data_fields.append(("timestamp_granularities[]", "segment"))
        data_fields.append(("timestamp_granularities[]", "word"))

        logger.info("Sending %s to Groq API (model=%s)", audio_file_path.name, model)

        with open(audio_file_path, "rb") as f:
            files = {
                "file": (audio_file_path.name, f),
            }
            resp = requests.post(
                url,
                headers=headers,
                data=data_fields,
                files=files,
                timeout=300,
            )

        if resp.status_code != 200:
            error_msg = f"Groq API error (status {resp.status_code}): {resp.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        raw = resp.json()

        # Parse duration
        try:
            info = sf.info(str(audio_path))
            duration = float(info.duration)
        except Exception:
            duration = float(raw.get("duration", 0.0))

        # Parse segments and word timestamps
        segments = []
        raw_segments = raw.get("segments", [])

        # Check for top-level words if returned by Groq
        all_words = raw.get("words", [])

        if not raw_segments and raw.get("text"):
            # Fallback if no segments provided
            segments.append({
                "id": 0,
                "start": 0.0,
                "end": round(duration, 3),
                "speaker": "SPEAKER_00",
                "text": raw.get("text", "").strip(),
                "words": [],
            })
        else:
            for i, seg in enumerate(raw_segments):
                seg_start = round(float(seg.get("start", 0.0)), 3)
                seg_end = round(float(seg.get("end", 0.0)), 3)

                seg_words = []
                if "words" in seg:
                    for w in seg["words"]:
                        seg_words.append({
                            "word": w.get("word", "").strip(),
                            "start": round(float(w.get("start", 0.0)), 3),
                            "end": round(float(w.get("end", 0.0)), 3),
                        })
                elif all_words:
                    for w in all_words:
                        w_start = round(float(w.get("start", 0.0)), 3)
                        w_end = round(float(w.get("end", 0.0)), 3)
                        if seg_start <= w_start <= seg_end or seg_start <= w_end <= seg_end:
                            seg_words.append({
                                "word": w.get("word", "").strip(),
                                "start": w_start,
                                "end": w_end,
                            })

                segments.append({
                    "id": i,
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": seg.get("speaker", "SPEAKER_00"),
                    "text": seg.get("text", "").strip(),
                    "words": seg_words,
                })

        detected_lang = raw.get("language", language)

        return {
            "status": "completed",
            "error": None,
            "metadata": {
                "audio_duration_seconds": round(duration, 2),
                "processing_time_seconds": round(time.time() - t0, 2),
                "inference_mode": "groq",
                "model": model,
                "language_detected": detected_lang,
                "num_speakers": len({s["speaker"] for s in segments}),
                "num_segments": len(segments),
            },
            "segments": segments,
        }

    def check_health(self) -> dict[str, Any]:
        """Check connection and API key validity with Groq."""
        import requests

        api_key = self._cfg.groq_api_key
        if not api_key:
            return {"status": "error", "message": "GROQ_API_KEY is not set"}

        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            resp = requests.get(GROQ_MODELS_URL, headers=headers, timeout=10)
            if resp.status_code == 200:
                models = [m["id"] for m in resp.json().get("data", []) if "whisper" in m.get("id", "")]
                return {
                    "status": "ok",
                    "provider": "groq",
                    "configured_model": self._cfg.groq_model,
                    "available_whisper_models": models,
                }
            else:
                return {
                    "status": "error",
                    "code": resp.status_code,
                    "message": resp.text,
                }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
