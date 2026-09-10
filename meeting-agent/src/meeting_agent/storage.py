"""Audio file storage.

``Storage`` is the seam between the API and wherever bytes actually live.
``LocalStorage`` is the only implementation today (a directory under
``OUTPUT_DIR``); swapping in object storage (S3/GCS) later means adding a new
``Storage`` implementation, not touching any caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Storage(ABC):
    @abstractmethod
    def save_audio(self, meeting_id: str, filename: str, data: bytes) -> str:
        """Persist audio bytes for a meeting and return a path/key that resolve() can read back."""

    @abstractmethod
    def resolve(self, path: str) -> Path:
        """Resolve a stored path/key to a local filesystem path for reading."""


class LocalStorage(Storage):
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def save_audio(self, meeting_id: str, filename: str, data: bytes) -> str:
        safe_name = Path(filename).name or "audio.webm"
        meeting_dir = self.base_dir / "audio" / meeting_id
        meeting_dir.mkdir(parents=True, exist_ok=True)
        dest = meeting_dir / safe_name
        dest.write_bytes(data)
        return str(dest)

    def resolve(self, path: str) -> Path:
        return Path(path)
