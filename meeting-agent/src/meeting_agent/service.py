"""Meeting lifecycle orchestration.

Implements the flow from the spec:

    audio -> transcribe_audio() -> save_transcript() -> generate_meeting_intelligence()
          -> save_meeting_intelligence() -> index_meeting_for_rag() -> mark_meeting_completed()

Each stage is idempotent (re-running it overwrites its own rows), so retrying a
meeting that failed at a given stage is just calling that stage's endpoint again
-- no separate "retry" endpoint or flag needed.

``transcribe`` and ``summarize`` are pulled in from the existing transcriber/
and summarizer/ packages, which is where Rushaan's (audio -> transcript) and
Arshia's (transcript -> intelligence) modules plug in. Today those packages
implement the work themselves (Whisper API/local, Anthropic/OpenAI/offline);
swapping in their services later means changing ``build_transcriber`` /
``build_summarizer`` internals, not this file.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from sqlalchemy.orm import Session

from . import dbmodels as m
from .config import Settings
from .models import Segment, Transcript
from .storage import Storage


class MeetingError(Exception):
    """A stage failed; the message is safe to return to API clients (no stack trace)."""


def create_meeting(db: Session, title: str | None) -> m.Meeting:
    meeting = m.Meeting(title=title or "Untitled Meeting", status=m.MeetingStatus.created)
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return meeting


def save_audio(db: Session, storage: Storage, meeting: m.Meeting, filename: str, data: bytes) -> m.Meeting:
    path = storage.save_audio(meeting.id, filename, data)
    meeting.audio_path = path
    meeting.status = m.MeetingStatus.uploaded
    meeting.error_message = None
    meeting.failed_stage = None
    db.commit()
    db.refresh(meeting)
    return meeting


def _fail(db: Session, meeting: m.Meeting, stage: str, exc: Exception) -> None:
    traceback.print_exc()  # server-side log only -- never handed to the client
    meeting.status = m.MeetingStatus.failed
    meeting.failed_stage = stage
    meeting.error_message = str(exc) or exc.__class__.__name__
    db.commit()


def run_transcription(db: Session, settings: Settings, meeting: m.Meeting) -> m.Meeting:
    if not meeting.audio_path:
        raise MeetingError("No audio uploaded for this meeting yet.")

    meeting.status = m.MeetingStatus.transcribing
    meeting.error_message = None
    meeting.failed_stage = None
    db.commit()

    try:
        from .transcriber.base import build_transcriber

        transcript = build_transcriber(settings).transcribe(Path(meeting.audio_path))
    except Exception as exc:
        _fail(db, meeting, "transcribing", exc)
        raise MeetingError(f"Transcription failed: {exc}") from exc

    for seg in list(meeting.segments):
        db.delete(seg)
    for p in list(meeting.participants):
        db.delete(p)
    db.flush()

    for s in transcript.segments:
        db.add(m.TranscriptSegment(meeting_id=meeting.id, start=s.start, end=s.end, speaker=s.speaker, text=s.text))
    for name in sorted({s.speaker for s in transcript.segments if s.speaker}):
        db.add(m.Participant(meeting_id=meeting.id, name=name))

    meeting.status = m.MeetingStatus.transcribed
    db.commit()
    db.refresh(meeting)
    return meeting


def run_processing(db: Session, settings: Settings, meeting: m.Meeting) -> m.Meeting:
    if not meeting.segments:
        raise MeetingError("No transcript to process yet -- call /transcribe first.")

    meeting.status = m.MeetingStatus.processing
    meeting.error_message = None
    meeting.failed_stage = None
    db.commit()

    transcript = Transcript(
        segments=[Segment(start=s.start, end=s.end, text=s.text, speaker=s.speaker) for s in meeting.segments]
    )

    try:
        from .summarizer.base import build_summarizer

        summary = build_summarizer(settings).summarize(transcript, title=meeting.title)
    except Exception as exc:
        _fail(db, meeting, "processing", exc)
        raise MeetingError(f"Processing failed: {exc}") from exc

    if meeting.summary is not None:
        db.delete(meeting.summary)
    for row in list(meeting.action_items):
        db.delete(row)
    for row in list(meeting.decisions):
        db.delete(row)
    for row in list(meeting.topics):
        db.delete(row)
    db.flush()

    db.add(
        m.Summary(
            meeting_id=meeting.id,
            executive_summary=summary.executive_summary,
            key_points_json=json.dumps([dp.summary for dp in summary.discussion_points if dp.summary]),
            open_questions_json=json.dumps(summary.open_questions),
            next_steps_json=json.dumps(summary.next_steps),
        )
    )
    for a in summary.action_items:
        db.add(m.ActionItem(meeting_id=meeting.id, task=a.task, owner=a.owner, due=a.due))
    for d in summary.decisions:
        db.add(m.Decision(meeting_id=meeting.id, text=d))
    for t in summary.key_topics:
        db.add(m.Topic(meeting_id=meeting.id, name=t))
    db.commit()

    # "Indexing" for RAG is Aman's module. Transcript segments are already
    # persisted and queryable, which is what /ask uses as a placeholder until
    # real vector indexing lands here.
    meeting.status = m.MeetingStatus.indexed
    db.commit()
    meeting.status = m.MeetingStatus.completed
    db.commit()
    db.refresh(meeting)
    return meeting


def ask(db: Session, question: str, meeting_id: str | None = None) -> tuple[str, list[tuple[str, float]]]:
    """Naive keyword search over stored transcript segments.

    Placeholder for Aman's RAG module: swap this function's body for real
    vector search + LLM answer synthesis. The /ask endpoint's request/response
    shape already matches the agreed RAG Answer contract, so that swap needs
    no API changes.
    """
    q_words = {w.strip(".,!?").lower() for w in question.split() if len(w) > 2}
    query = db.query(m.TranscriptSegment)
    if meeting_id:
        query = query.filter(m.TranscriptSegment.meeting_id == meeting_id)

    scored: list[tuple[int, m.TranscriptSegment]] = []
    for seg in query.all():
        overlap = len(q_words & {w.strip(".,!?").lower() for w in seg.text.split()})
        if overlap:
            scored.append((overlap, seg))
    scored.sort(key=lambda t: -t[0])
    top = scored[:3]

    if not top:
        return "I couldn't find anything relevant in the indexed meetings yet.", []

    answer = " ".join(seg.text for _, seg in top)
    sources = [(seg.meeting_id, seg.start) for _, seg in top]
    return answer, sources
