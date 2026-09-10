"""FastAPI backend -- the API surface the spec describes, wired to the existing
transcribe/summarize pipeline and a persisted meeting lifecycle.

Run it with:

    uv run uvicorn meeting_agent.app:app --reload

or ``uv run python -m meeting_agent serve``.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from sqlalchemy.orm import Session

from . import dbmodels as m
from . import schemas as sch
from . import service
from .config import Settings
from .db import get_db, init_db
from .storage import LocalStorage


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Meeting Agent API", version="0.1.0", lifespan=_lifespan)


def get_settings() -> Settings:
    return Settings()


def _meeting_out(meeting: m.Meeting) -> sch.MeetingOut:
    status = meeting.status.value if hasattr(meeting.status, "value") else meeting.status
    return sch.MeetingOut(
        id=meeting.id,
        title=meeting.title,
        status=status,
        failed_stage=meeting.failed_stage,
        error_message=meeting.error_message,
        created_at=meeting.created_at.isoformat(),
        updated_at=meeting.updated_at.isoformat(),
    )


def _get_meeting_or_404(db: Session, meeting_id: str) -> m.Meeting:
    meeting = db.get(m.Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(404, "Meeting not found")
    return meeting


@app.post("/meetings", response_model=sch.MeetingOut)
def create_meeting(body: sch.MeetingCreate, db: Session = Depends(get_db)) -> sch.MeetingOut:
    meeting = service.create_meeting(db, body.title)
    return _meeting_out(meeting)


@app.get("/meetings", response_model=list[sch.MeetingOut])
def list_meetings(db: Session = Depends(get_db)) -> list[sch.MeetingOut]:
    rows = db.query(m.Meeting).order_by(m.Meeting.created_at.desc()).all()
    return [_meeting_out(r) for r in rows]


@app.get("/meetings/{meeting_id}", response_model=sch.MeetingOut)
def get_meeting(meeting_id: str, db: Session = Depends(get_db)) -> sch.MeetingOut:
    return _meeting_out(_get_meeting_or_404(db, meeting_id))


@app.post("/meetings/{meeting_id}/audio", response_model=sch.MeetingOut)
async def upload_audio(
    meeting_id: str,
    file: UploadFile,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> sch.MeetingOut:
    meeting = _get_meeting_or_404(db, meeting_id)
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded audio file is empty")
    storage = LocalStorage(settings.output_dir)
    meeting = service.save_audio(db, storage, meeting, file.filename or "audio.webm", data)
    return _meeting_out(meeting)


@app.post("/meetings/{meeting_id}/transcribe", response_model=sch.MeetingOut)
def transcribe_meeting(
    meeting_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> sch.MeetingOut:
    meeting = _get_meeting_or_404(db, meeting_id)
    try:
        meeting = service.run_transcription(db, settings, meeting)
    except service.MeetingError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _meeting_out(meeting)


@app.post("/meetings/{meeting_id}/process", response_model=sch.MeetingOut)
def process_meeting(
    meeting_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> sch.MeetingOut:
    meeting = _get_meeting_or_404(db, meeting_id)
    try:
        meeting = service.run_processing(db, settings, meeting)
    except service.MeetingError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _meeting_out(meeting)


@app.get("/meetings/{meeting_id}/transcript", response_model=sch.TranscriptOut)
def get_transcript(meeting_id: str, db: Session = Depends(get_db)) -> sch.TranscriptOut:
    meeting = _get_meeting_or_404(db, meeting_id)
    return sch.TranscriptOut(
        meeting_id=meeting.id,
        segments=[sch.SegmentOut(start=s.start, end=s.end, text=s.text, speaker=s.speaker) for s in meeting.segments],
    )


@app.get("/meetings/{meeting_id}/summary", response_model=sch.SummaryOut)
def get_summary(meeting_id: str, db: Session = Depends(get_db)) -> sch.SummaryOut:
    meeting = _get_meeting_or_404(db, meeting_id)
    if meeting.summary is None:
        raise HTTPException(404, "No summary yet -- call /process first")
    return sch.SummaryOut(
        meeting_id=meeting.id,
        summary=meeting.summary.executive_summary,
        key_points=json.loads(meeting.summary.key_points_json),
        decisions=[d.text for d in meeting.decisions],
        action_items=[sch.ActionItemOut(task=a.task, owner=a.owner, due=a.due) for a in meeting.action_items],
        topics=[t.name for t in meeting.topics],
        open_questions=json.loads(meeting.summary.open_questions_json),
    )


@app.get("/meetings/{meeting_id}/action-items", response_model=list[sch.ActionItemOut])
def get_action_items(meeting_id: str, db: Session = Depends(get_db)) -> list[sch.ActionItemOut]:
    meeting = _get_meeting_or_404(db, meeting_id)
    return [sch.ActionItemOut(task=a.task, owner=a.owner, due=a.due) for a in meeting.action_items]


@app.post("/ask", response_model=sch.AskResponse)
def ask(body: sch.AskRequest, db: Session = Depends(get_db)) -> sch.AskResponse:
    answer, sources = service.ask(db, body.question, body.meeting_id)
    return sch.AskResponse(
        answer=answer,
        sources=[sch.AskSource(meeting_id=mid, timestamp=ts) for mid, ts in sources],
    )
