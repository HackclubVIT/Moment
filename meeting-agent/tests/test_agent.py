import json
from pathlib import Path

from meeting_agent.agent import Agent
from meeting_agent.config import Settings


class FakeRecorder:
    def __init__(self):
        self.recording = False

    def status(self):
        return {"recording": self.recording}


class FakePipeline:
    def __init__(self):
        self.recorder = FakeRecorder()
        self.audio_path = None
        self.transcript = None
        self.summary = None
        self.mom_path = None
        self.joined = None

    def join(self, url):
        self.joined = url
        return {"ok": True, "message": f"Auto-joined {url}"}

    def start_recording(self, mode="tab"):
        self.recorder.recording = True
        return f"Recording started ({mode} mode)."

    def stop_recording(self):
        self.recorder.recording = False
        self.audio_path = Path("output/recording.webm")
        return "Recording saved to output/recording.webm"

    def transcribe(self):
        from meeting_agent.models import Transcript

        self.transcript = Transcript(segments=[])
        return "Transcribed 5 segment(s)."

    def summarize(self):
        from meeting_agent.models import MeetingSummary

        self.summary = MeetingSummary(executive_summary="We shipped the onboarding flow.")
        return "Summary ready: 2 decision(s), 3 action item(s)."

    def write_mom(self, title=None):
        self.mom_path = Path("output/MOM_demo.md")
        return f"MOM saved to {self.mom_path}"

    def status_text(self):
        return "platform: google-meet\nrecording: inactive"


def make_agent() -> tuple[Agent, FakePipeline]:
    settings = Settings(llm_engine="offline")
    pipeline = FakePipeline()
    return Agent(settings, pipeline), pipeline


def test_rule_reply_join_url():
    agent, pipeline = make_agent()
    reply = agent.reply("please join https://meet.google.com/abc-defg-hij and take minutes")
    assert reply.startswith("Joined the meeting")
    assert pipeline.joined == "https://meet.google.com/abc-defg-hij"
    assert pipeline.recorder.recording is True


def test_rule_reply_stop_flow():
    agent, pipeline = make_agent()
    pipeline.audio_path = Path("output/recording.webm")
    pipeline.recorder.recording = True
    reply = agent.reply("stop, the meeting is over")
    assert "MOM saved" in reply
    assert pipeline.mom_path is not None
    assert pipeline.recorder.recording is False


def test_rule_reply_help():
    agent, _ = make_agent()
    assert "join" in agent.reply("help")
    assert "join" in agent.reply("")


def test_command_join():
    agent, pipeline = make_agent()
    agent.reply("/join https://zoom.us/j/123456789")
    assert pipeline.joined == "https://zoom.us/j/123456789"


def test_tool_dispatch_status():
    agent, _ = make_agent()
    out = json.loads(agent._run_tool("status", {}))
    assert out["ok"] is True
    assert "google-meet" in out["status"]


def test_tool_dispatch_unknown():
    agent, _ = make_agent()
    out = json.loads(agent._run_tool("nope", {}))
    assert out["ok"] is False
