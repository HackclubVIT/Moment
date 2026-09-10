"""Parse and serialize transcripts from plain-text files and STT output."""

from __future__ import annotations

import re
from pathlib import Path

from .models import Segment, Transcript

# Matches a leading "[mm:ss]" / "[hh:mm:ss]" timestamp.
_TS_START = re.compile(r"^\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?\s+")

# Matches "Speaker: text" at the start of a line (optionally after a timestamp).
_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z .'’-]{0,24}?):\s+(.+)$")


def _parse_ts(groups: tuple[str, str, str | None]) -> float:
    h, m, s = groups
    return int(h) * 60 + int(m) + int(s or 0)


def parse_transcript_text(text: str, path: str | None = None, language: str | None = None) -> Transcript:
    """Parse lines like '[0:12] Alice: hello there' into timestamped segments.

    Lines without a speaker prefix are appended as plain-text continuation.
    """
    transcript = Transcript(path=path, language=language)
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        start = 0.0
        ts = _TS_START.match(line)
        if ts:
            start = _parse_ts(ts.groups())
            line = line[ts.end() :].strip()
        m = _LINE_RE.match(line)
        if m:
            transcript.segments.append(
                Segment(start=start, end=start, speaker=m.group(1).strip(), text=m.group(2).strip())
            )
        else:
            transcript.append_plain_text(line)
    return transcript


def load_transcript(path: str | Path) -> Transcript:
    path = Path(path)
    return parse_transcript_text(path.read_text(encoding="utf-8"), path=str(path))


def save_transcript_text(transcript: Transcript, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for s in transcript.segments:
        ts = f"[{int(s.start // 60):02d}:{int(s.start % 60):02d}] " if s.start else ""
        speaker = f"{s.speaker}: " if s.speaker else ""
        lines.append(f"{ts}{speaker}{s.text}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
