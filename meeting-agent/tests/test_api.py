"""API tests for the backend: create meeting, upload audio, process, get results,
ask a question, and the failure/retry path -- the cases called out in the spec.

The real transcriber (Whisper API/local) and the network LLM summarizers are not
exercised here: transcription is monkeypatched to a fake, and the offline
extractive summarizer (no API key, no network) stands in for the LLM one so
these tests run anywhere with no external dependencies.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meeting_agent.app import app, get_settings
from meeting_agent.config import Settings
from meeting_agent.db import configure_engine
from meeting_agent.models import Segment, Transcript

FAKE_SEGMENTS = [
    Segment(start=0.0, end=5.0, speaker="Alice", text="We decided to ship the onboarding flow this week."),
    Segment(start=5.0, end=10.0, speaker="Bob", text="Bob will send the deployment checklist by Friday."),
    Segment(start=10.0, end=15.0, speaker="Alice", text="Should we discuss the rollback plan too?"),
]


class FakeTranscriber:
    def __init__(self, segments=None):
        self.segments = segments if segments is not None else FAKE_SEGMENTS

    def transcribe(self, audio_path):
        return Transcript(segments=list(self.segments))


class FailingTranscriber:
    def transcribe(self, audio_path):
        raise RuntimeError("STT backend unreachable")


@pytest.fixture()
def client(tmp_path):
    test_settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        output_dir=tmp_path / "output",
        llm_engine="offline",
    )
    # Repoints the app's single DB engine at an isolated tmp file for this test,
    # so running the suite never touches the real DATABASE_URL/meeting_agent.db.
    configure_engine(test_settings)
    app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def _create_meeting(client, title="Sprint Planning") -> str:
    resp = client.post("/meetings", json={"title": title})
    assert resp.status_code == 200
    return resp.json()["id"]


def _upload_audio(client, meeting_id: str) -> None:
    resp = client.post(
        f"/meetings/{meeting_id}/audio",
        files={"file": ("recording.webm", b"not-really-audio-but-nonempty", "audio/webm")},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "uploaded"


def test_create_meeting(client):
    resp = client.post("/meetings", json={"title": "Weekly Sync"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Weekly Sync"
    assert body["status"] == "created"
    assert body["id"]


def test_get_meeting_not_found(client):
    resp = client.get("/meetings/does-not-exist")
    assert resp.status_code == 404


def test_upload_audio(client):
    meeting_id = _create_meeting(client)
    _upload_audio(client, meeting_id)
    resp = client.get(f"/meetings/{meeting_id}")
    assert resp.json()["status"] == "uploaded"


def test_upload_audio_rejects_empty_file(client):
    meeting_id = _create_meeting(client)
    resp = client.post(f"/meetings/{meeting_id}/audio", files={"file": ("empty.webm", b"", "audio/webm")})
    assert resp.status_code == 400


def test_full_pipeline_and_get_endpoints(client, monkeypatch):
    monkeypatch.setattr("meeting_agent.transcriber.base.build_transcriber", lambda settings: FakeTranscriber())

    meeting_id = _create_meeting(client, title="Launch Review")
    _upload_audio(client, meeting_id)

    resp = client.post(f"/meetings/{meeting_id}/transcribe")
    assert resp.status_code == 200
    assert resp.json()["status"] == "transcribed"

    resp = client.get(f"/meetings/{meeting_id}/transcript")
    assert resp.status_code == 200
    segments = resp.json()["segments"]
    assert len(segments) == 3
    assert segments[1]["speaker"] == "Bob"

    resp = client.post(f"/meetings/{meeting_id}/process")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"

    resp = client.get(f"/meetings/{meeting_id}/summary")
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["meeting_id"] == meeting_id
    assert summary["decisions"]

    resp = client.get(f"/meetings/{meeting_id}/action-items")
    assert resp.status_code == 200
    tasks = [a["task"] for a in resp.json()]
    assert any("checklist" in t.lower() for t in tasks)


def test_process_without_transcript_fails(client):
    meeting_id = _create_meeting(client)
    resp = client.post(f"/meetings/{meeting_id}/process")
    assert resp.status_code == 422
    assert "transcript" in resp.json()["detail"].lower()


def test_summary_before_process_is_404(client):
    meeting_id = _create_meeting(client)
    resp = client.get(f"/meetings/{meeting_id}/summary")
    assert resp.status_code == 404


def test_ask_returns_relevant_source(client, monkeypatch):
    monkeypatch.setattr("meeting_agent.transcriber.base.build_transcriber", lambda settings: FakeTranscriber())
    meeting_id = _create_meeting(client)
    _upload_audio(client, meeting_id)
    client.post(f"/meetings/{meeting_id}/transcribe")
    client.post(f"/meetings/{meeting_id}/process")

    resp = client.post("/ask", json={"question": "Who will send the deployment checklist?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert any(s["meeting_id"] == meeting_id for s in body["sources"])


def test_ask_with_no_matching_meetings(client):
    resp = client.post("/ask", json={"question": "anything at all"})
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


def test_failed_transcription_then_retry(client, monkeypatch):
    meeting_id = _create_meeting(client)
    _upload_audio(client, meeting_id)

    monkeypatch.setattr("meeting_agent.transcriber.base.build_transcriber", lambda settings: FailingTranscriber())
    resp = client.post(f"/meetings/{meeting_id}/transcribe")
    assert resp.status_code == 422
    assert "traceback" not in resp.json()["detail"].lower()

    meeting = client.get(f"/meetings/{meeting_id}").json()
    assert meeting["status"] == "failed"
    assert meeting["failed_stage"] == "transcribing"
    assert meeting["error_message"]

    # Retry: fix the backend and call the same endpoint again -- no separate retry API.
    monkeypatch.setattr("meeting_agent.transcriber.base.build_transcriber", lambda settings: FakeTranscriber())
    resp = client.post(f"/meetings/{meeting_id}/transcribe")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "transcribed"
    assert body["error_message"] is None
    assert body["failed_stage"] is None


def test_transcribe_without_audio_fails_cleanly(client):
    meeting_id = _create_meeting(client)
    resp = client.post(f"/meetings/{meeting_id}/transcribe")
    assert resp.status_code == 422
    assert "audio" in resp.json()["detail"].lower()


def test_list_meetings(client):
    _create_meeting(client, title="One")
    _create_meeting(client, title="Two")
    resp = client.get("/meetings")
    assert resp.status_code == 200
    titles = {m["title"] for m in resp.json()}
    assert {"One", "Two"} <= titles
