"""Meeting joining: platform detection + Playwright joiners."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://[^\s]+")
_BARE_RE = re.compile(
    r"(?P<url>(?:meet\.google\.com/[a-z0-9-]+|"
    r"zoom\.us/j/\d+|"
    r"teams\.microsoft\.com/l/meetup-join/[\w-]+))",
    re.I,
)


def detect_platform(url: str) -> str | None:
    """Identify the meeting platform from a URL: 'google-meet' | 'zoom' | 'teams' | None."""
    u = (url or "").lower()
    if "meet.google.com" in u:
        return "google-meet"
    if "zoom.us" in u and ("/j/" in u or "/wc/" in u):
        return "zoom"
    if ("teams.microsoft.com" in u and "/l/meetup-join/" in u) or "teams.live.com" in u:
        return "teams"
    return None


def find_meeting_url(text: str) -> str | None:
    """Extract a meeting URL from free-form user text (with or without scheme)."""
    for m in _URL_RE.finditer(text or ""):
        url = m.group(0).rstrip(".,;)>\"'")
        if detect_platform(url) is not None:
            return url
    for m in _BARE_RE.finditer(text or ""):
        url = m.group("url").rstrip(".,;)>\"'")
        if not url.startswith("http"):
            url = "https://" + url
        return url
    return None
