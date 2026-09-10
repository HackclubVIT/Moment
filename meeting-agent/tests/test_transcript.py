from meeting_agent.parsing import parse_transcript_text, save_transcript_text


def test_parse_with_speakers_and_timestamps():
    t = parse_transcript_text(
        "[0:12] Alice: hello there\n[1:30] Bob: let's go\njust a continuation line\n"
    )
    assert len(t.segments) == 3
    assert t.segments[0].speaker == "Alice"
    assert t.segments[0].text == "hello there"
    assert t.segments[0].start == 12.0
    assert t.segments[1].speaker == "Bob"
    assert t.segments[1].start == 90.0
    assert t.segments[2].speaker is None


def test_parse_plain_text():
    t = parse_transcript_text("This meeting discussed the roadmap.\nWe agreed on next steps.")
    assert len(t.segments) == 2
    assert "roadmap" in t.text


def test_save_and_reload(tmp_path):
    t = parse_transcript_text("[0:00] Alice: hi")
    path = save_transcript_text(t, tmp_path / "t.txt")
    assert path.exists()
    from meeting_agent.parsing import load_transcript

    t2 = load_transcript(path)
    assert t2.segments[0].speaker == "Alice"
