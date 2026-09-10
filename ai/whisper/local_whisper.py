"""Local WhisperX fallback — runs the full pipeline on local hardware.

Used when the cloud (Groq) API is unreachable, rate limited, or when offline.
Auto-detects CUDA GPU vs CPU and adjusts compute type and model size accordingly.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from .config import WhisperConfig, load_config

logger = logging.getLogger(__name__)


class LocalWhisperX:
    """Local WhisperX inference engine.

    Models are lazy-loaded on first ``transcribe()`` call to avoid heavy
    startup cost when the class is merely imported.
    """

    def __init__(self, config: WhisperConfig | None = None) -> None:
        self._cfg = config or load_config()
        self._model = None
        self._device: str | None = None
        self._compute_type: str | None = None

    def _resolve_device(self) -> tuple[str, str]:
        """Detect best device and matching compute type."""
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            compute = "float16"
        else:
            device = "cpu"
            compute = "int8"

        if self._cfg.local_compute_type != "auto":
            compute = self._cfg.local_compute_type

        return device, compute

    def _ensure_model(self) -> None:
        """Lazy-load the Whisper model if not yet loaded."""
        if self._model is not None:
            return

        import whisperx

        self._device, self._compute_type = self._resolve_device()

        # On CPU, prefer a smaller model for reasonable speed
        model_size = self._cfg.local_model_size
        if self._device == "cpu" and model_size == "large-v3":
            logger.warning(
                "large-v3 on CPU is very slow; consider setting "
                "WHISPER_LOCAL_MODEL=base or medium in .env"
            )

        logger.info(
            "Loading WhisperX model=%s device=%s compute=%s",
            model_size,
            self._device,
            self._compute_type,
        )
        self._model = whisperx.load_model(
            model_size,
            device=self._device,
            compute_type=self._compute_type,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "auto",
        translate: bool = True,
    ) -> dict:
        """Run the full WhisperX pipeline locally.

        Args:
            audio_path: Path to a WAV audio file.
            language: ISO language code or ``"auto"`` for detection.
            translate: If True, use Whisper's translate task for non-English.

        Returns:
            Dict with ``status``, ``metadata``, ``segments`` matching the
            project's transcript JSON schema.
        """
        import soundfile as sf
        import whisperx

        self._ensure_model()

        audio_path = str(audio_path)
        t0 = time.time()

        audio = whisperx.load_audio(audio_path)

        # 1. Transcribe
        use_translation = translate and language not in ("auto", "en")
        task = "translate" if use_translation else "transcribe"
        result = self._model.transcribe(
            audio,
            batch_size=16 if self._device == "cuda" else 4,
            language=None if language == "auto" else language,
            task=task,
        )
        detected_lang = result.get("language", language)

        # 2. Forced alignment
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang,
                device=self._device,
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device=self._device,
                return_char_alignments=False,
            )
        except Exception:
            pass  # alignment not available for all languages

        # 3. Diarization
        hf_token = self._cfg.hf_token or os.environ.get("HF_TOKEN", "")
        if hf_token:
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=self._device,
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as exc:
                logger.warning("Diarization encountered an error (continuing without diarization): %s", exc)

        # 4. Build result
        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            segments.append({
                "id": i,
                "start": round(seg.get("start", 0.0), 3),
                "end": round(seg.get("end", 0.0), 3),
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "text": seg.get("text", "").strip(),
                "words": [
                    {
                        "word": w.get("word", ""),
                        "start": round(w.get("start", 0.0), 3),
                        "end": round(w.get("end", 0.0), 3),
                    }
                    for w in seg.get("words", [])
                ],
            })

        info = sf.info(audio_path)

        return {
            "status": "completed",
            "error": None,
            "metadata": {
                "audio_duration_seconds": round(info.duration, 2),
                "processing_time_seconds": round(time.time() - t0, 2),
                "inference_mode": "local",
                "model": self._cfg.local_model_size,
                "language_detected": detected_lang,
                "num_speakers": len({s["speaker"] for s in segments}),
                "num_segments": len(segments),
            },
            "segments": segments,
        }
