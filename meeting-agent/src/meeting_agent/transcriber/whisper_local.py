"""Local, offline speech-to-text via faster-whisper (ctranslate2)."""

from __future__ import annotations

from pathlib import Path

from ..models import Segment, Transcript
from .base import Transcriber


class WhisperLocal(Transcriber):
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "faster-whisper is not installed. Run: uv sync --extra local-stt"
            ) from exc
        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )

    def transcribe(self, audio_path) -> Transcript:
        self._ensure_model()
        segments_iter, info = self._model.transcribe(str(audio_path))
        segments: list[Segment] = []
        for s in segments_iter:
            text = (s.text or "").strip()
            if not text:
                continue
            segments.append(Segment(start=float(s.start), end=float(s.end), text=text))
        return Transcript(
            path=str(Path(audio_path)),
            language=getattr(info, "language", None),
            segments=segments,
        )
