"""Best-effort auto-join logic for Google Meet, Zoom web, and Microsoft Teams web."""

from __future__ import annotations

import time

from .detect import detect_platform


def join_meeting(page, url: str, platform: str | None = None) -> dict:
    """Open the meeting URL and try to click the join button.

    Returns {"ok": bool, "platform": str|None, "message": str}. If auto-join fails
    the browser window stays open for the user to join manually.
    """
    platform = platform or detect_platform(url)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
    except Exception as exc:
        return {"ok": False, "platform": platform, "message": f"Could not open {url}: {exc}"}

    if platform == "google-meet":
        return _join_google_meet(page)
    if platform == "zoom":
        return _join_zoom(page)
    if platform == "teams":
        return _join_teams(page)

    time.sleep(8)
    return {
        "ok": True,
        "platform": None,
        "message": "Opened the URL in the browser. Please join the meeting manually if needed.",
    }


def _click_first(page, names: list[str], timeout_ms: int = 3500) -> bool:
    """Try role-based buttons by accessible name; return True if one was clicked."""
    for name in names:
        try:
            btn = page.get_by_role("button", name=name, exact=False).first
            btn.wait_for(state="visible", timeout=timeout_ms)
            btn.click(timeout=8000)
            return True
        except Exception:
            continue
    return False


def _join_google_meet(page) -> dict:
    try:
        page.context.grant_permissions(["microphone", "camera"], origin="https://meet.google.com")
    except Exception:
        pass
    page.wait_for_timeout(6000)

    # Dismiss the first-run "Got it" bubble if present.
    _click_first(page, ["Got it", "Maybe later", "Dismiss"], timeout_ms=2500)

    clicked = _click_first(page, ["Join now", "Ask to join"], timeout_ms=6000)
    page.wait_for_timeout(5000)
    if clicked:
        return {
            "ok": True,
            "platform": "google-meet",
            "message": "Auto-joined Google Meet. If the mic/camera prompt appeared, it was handled.",
        }
    return {
        "ok": False,
        "platform": "google-meet",
        "message": "Could not find the 'Join now' button. The browser is open — click it manually.",
    }


def _join_zoom(page) -> dict:
    page.wait_for_timeout(6000)
    _click_first(page, ["Launch Meeting", "Join from Your Browser", "Join from your browser", "Join"])
    page.wait_for_timeout(4000)
    _click_first(page, ["Join with Computer Audio", "Join Audio by Computer", "Join", "Continue"])
    page.wait_for_timeout(3000)
    return {
        "ok": True,
        "platform": "zoom",
        "message": "Zoom: clicked the join buttons. If the meeting window did not open, join manually.",
    }


def _join_teams(page) -> dict:
    page.wait_for_timeout(7000)
    _click_first(page, ["Use the web app instead", "Continue on this browser", "Join now", "Join"])
    page.wait_for_timeout(3000)
    return {
        "ok": True,
        "platform": "teams",
        "message": "Teams: clicked the join buttons. If a consent dialog appears, click it manually.",
    }
