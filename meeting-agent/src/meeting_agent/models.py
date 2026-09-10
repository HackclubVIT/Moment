"""Shared data models: transcript segments and the structured meeting summary (MOM content)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A single utterance with optional timestamp and speaker label."""

    start: float = 0.0
    end: float = 0.0
    text: str = ""
    speaker: str | None = None

    def __str__(self) -> str:
        prefix = f"{self.speaker}: " if self.speaker else ""
        return f"{prefix}{self.text}"


class Transcript(BaseModel):
    """The full speech-to-text output for one meeting."""

    path: str | None = None
    language: str | None = None
    segments: list[Segment] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(str(s) for s in self.segments)

    def append_plain_text(self, text: str) -> None:
        if text.strip():
            self.segments.append(Segment(text=text.strip()))


class DiscussionPoint(BaseModel):
    topic: str
    summary: str


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    due: str | None = None


class MeetingSummary(BaseModel):
    """Structured minutes-of-meeting content."""

    title: str = "Untitled Meeting"
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    attendees: list[str] = Field(default_factory=list)
    executive_summary: str = ""
    key_topics: list[str] = Field(default_factory=list)
    discussion_points: list[DiscussionPoint] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
