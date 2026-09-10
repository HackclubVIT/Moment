"""Minimal transcript cleaning.

Only applies safe, lossless transformations so that the RAG module (Aman)
gets segments as close to the original speech as possible.
"""

from __future__ import annotations

import re


def clean_transcript(segments: list[dict]) -> list[dict]:
    """Apply minimal cleaning to transcript segments.

    Operations (in order):
        1. Strip leading/trailing whitespace from text
        2. Normalize internal whitespace (multiple spaces → single)
        3. Remove empty segments (empty or whitespace-only text)
        4. Remove segments with invalid timestamps (start >= end)
        5. Collapse duplicate consecutive segments (same text + same speaker)
        6. Re-index segment IDs sequentially

    Args:
        segments: List of segment dicts with at least ``text``, ``start``,
                  ``end``, and ``speaker`` keys.

    Returns:
        A new list of cleaned segments (the input is not mutated).
    """
    cleaned: list[dict] = []

    for seg in segments:
        text = seg.get("text", "")

        # 1 & 2: strip + normalize internal whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 3: skip empty segments
        if not text:
            continue

        # 4: validate timestamps
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        if start >= end:
            continue

        # 5: collapse duplicate consecutive segments
        if cleaned:
            prev = cleaned[-1]
            if prev["text"] == text and prev.get("speaker") == seg.get("speaker"):
                # Extend the previous segment's end time
                prev["end"] = max(prev["end"], end)
                continue

        cleaned.append({
            **seg,
            "text": text,
            "start": start,
            "end": end,
        })

    # 6: re-index IDs
    for i, seg in enumerate(cleaned):
        seg["id"] = i

    return cleaned
