"""API request/response schemas — shaped to match the contracts agreed in the spec
(Transcript, Meeting Intelligence, RAG Answer)."""

from __future__ import annotations

from pydantic import BaseModel


class MeetingCreate(BaseModel):
    title: str | None = None


class MeetingOut(BaseModel):
    id: str
    title: str
    status: str
    failed_stage: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class TranscriptOut(BaseModel):
    meeting_id: str
    segments: list[SegmentOut]


class ActionItemOut(BaseModel):
    task: str
    owner: str | None = None
    due: str | None = None


class SummaryOut(BaseModel):
    meeting_id: str
    summary: str
    key_points: list[str]
    decisions: list[str]
    action_items: list[ActionItemOut]
    topics: list[str]
    open_questions: list[str]


class AskRequest(BaseModel):
    question: str
    meeting_id: str | None = None


class AskSource(BaseModel):
    meeting_id: str
    timestamp: float


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource]
