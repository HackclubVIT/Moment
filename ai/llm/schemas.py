from pydantic import BaseModel, Field
from typing import List


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    deadline: str | None = None


class Decision(BaseModel):
    decision: str
    context: str | None = None


class MeetingIntelligence(BaseModel):
    summary: str
    key_points: List[str] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)