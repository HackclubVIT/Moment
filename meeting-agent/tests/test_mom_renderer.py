from meeting_agent.models import ActionItem, DiscussionPoint, MeetingSummary
from meeting_agent.mom.renderer import render_mom, save_mom


def _summary() -> MeetingSummary:
    return MeetingSummary(
        title="Sprint Review",
        attendees=["Alice", "Bob"],
        executive_summary="Reviewed the onboarding flow and the login bug.",
        key_topics=["Onboarding redesign", "Login bug"],
        discussion_points=[DiscussionPoint(topic="Onboarding", summary="Discussed the new flow.")],
        decisions=["Ship the onboarding redesign by Friday."],
        action_items=[
            ActionItem(task="Draft API spec", owner="Alice", due="Wednesday"),
            ActionItem(task="Fix login bug", owner="Carol", due=None),
        ],
        open_questions=["Database migration timing?"],
        next_steps=["Draft API spec", "Fix login bug"],
    )


def test_render_contains_sections():
    md = render_mom(_summary())
    assert "# Minutes of Meeting" in md
    assert "## Executive Summary" in md
    assert "## Decisions" in md
    assert "## Action Items" in md
    assert "| # | Task | Owner | Due |" in md
    assert "## Open Questions / Risks" in md
    assert "Alice" in md
    assert "Wednesday" in md


def test_save_mom(tmp_path):
    path = save_mom(_summary(), tmp_path, title="Sprint Review")
    assert path.exists()
    assert "sprint-review" in path.name
    assert path.read_text(encoding="utf-8").startswith("# Minutes of Meeting")
