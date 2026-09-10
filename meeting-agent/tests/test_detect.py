from meeting_agent.meeting.detect import detect_platform, find_meeting_url


def test_detect_platform():
    assert detect_platform("https://meet.google.com/abc-defg-hij") == "google-meet"
    assert detect_platform("https://zoom.us/j/123456789?pwd=xyz") == "zoom"
    assert detect_platform("https://teams.microsoft.com/l/meetup-join/19%3ameeting%40thread.v2") == "teams"
    assert detect_platform("https://example.com/not-a-meeting") is None


def test_find_meeting_url_in_text():
    text = "Hey, can you join https://meet.google.com/abc-defg-hij and take minutes?"
    assert find_meeting_url(text) == "https://meet.google.com/abc-defg-hij"


def test_find_bare_meet_url():
    assert find_meeting_url("join meet.google.com/xyz-123 now") == "https://meet.google.com/xyz-123"


def test_find_none():
    assert find_meeting_url("no meetings here") is None
