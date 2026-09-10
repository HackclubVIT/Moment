"""Meeting joining (browser automation)."""

from .detect import detect_platform, find_meeting_url
from .joiners import join_meeting

__all__ = ["detect_platform", "find_meeting_url", "join_meeting"]
