from pathlib import Path

from meeting_agent.parsing import load_transcript
from meeting_agent.summarizer.extractive import ExtractiveSummarizer

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_transcript.txt"


def test_extractive_finds_decisions_and_action_items():
    transcript = load_transcript(SAMPLE)
    summary = ExtractiveSummarizer().summarize(transcript, title="Sprint Review")

    assert summary.title == "Sprint Review"
    assert summary.decisions, "expected at least one decision"
    assert any("onboarding" in d.lower() for d in summary.decisions)
    assert summary.action_items, "expected at least one action item"
    assert summary.executive_summary
    assert summary.key_topics


def test_action_item_owner_and_due():
    transcript = load_transcript(SAMPLE)
    summary = ExtractiveSummarizer().summarize(transcript)
    tasks = " ".join(a.task for a in summary.action_items)
    assert "API spec" in tasks
    owner_by_task = {a.task: a.owner for a in summary.action_items}
    spec = next(a for a in summary.action_items if "API spec" in a.task)
    assert spec.owner == "Alice", f"expected Alice as owner, got {spec.owner}"
    assert spec.due is not None and "Wednesday" in spec.due


def test_attendees():
    transcript = load_transcript(SAMPLE)
    summary = ExtractiveSummarizer().summarize(transcript)
    assert summary.attendees == ["Alice", "Bob", "Carol"]


def test_open_questions():
    transcript = load_transcript(SAMPLE)
    summary = ExtractiveSummarizer().summarize(transcript)
    assert any("database migration" in q.lower() for q in summary.open_questions)


def test_empty_transcript():
    from meeting_agent.models import Transcript

    summary = ExtractiveSummarizer().summarize(Transcript())
    assert summary.decisions == []
    assert summary.action_items == []
