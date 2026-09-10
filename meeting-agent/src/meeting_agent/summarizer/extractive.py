"""Offline, dependency-free extractive summarizer.

Heuristics used:
- Cluster transcript segments by silence gaps to find discussion topics.
- Score sentences by term frequency to pick the most salient ones.
- Label topics with the top content keywords.
- Detect decisions / action items / open questions with cue phrases.
- Attribute action-item owners from speaker labels or "Name will ..." phrasing.
"""

from __future__ import annotations

import re
from collections import Counter

from ..models import ActionItem, DiscussionPoint, MeetingSummary, Segment
from ..parsing import Transcript
from .base import Summarizer

_STOPWORDS = set(
    """a an the and or but if then so of to in on at for with as by from is are was were be
    been being it its this that these those i we you he she they me my our your his her their
    them not no don't do does did have has had can could will would should shall may might must
    about into over under after before between during without against up down out off again
    further once here there all any both each few more most other some such only own same too
    very just because until while per than etc""".split()
)

_MEETING_WORDS = set(
    """meeting discussion talk talking speak speaking say said says going wanna want need think
    thought thing things yeah okay ok um uh erm like actually basically right well yes just
    really sure good great thanks thank guys everyone everybody hello hi welcome alright cool
    perfect point line call today tomorrow week month time minute minutes hour hours work people
    team project product issue question answer answers next let's lets know make makes made get
    got take took give gave put set see saw look looking come came back forward please""".split()
)

_DECISION = re.compile(
    r"\b(decide[ds]?|decision|agreed?\b|approved?\b|confirm[ed]?\b|settl[ed]?\b|"
    r"we'?ll go with|we will go with|going with|let'?s go with|finaliz[ed]?\b|"
    r"resolved?\b|sign(?:ed)? off|locked in)\b",
    re.I,
)
_ACTION = re.compile(r"\b(action item|action items|follow[- ]?up|to[- ]?do|todo|next steps?|owner)\b", re.I)
_FUTURE = re.compile(
    r"\b(will|should|need to|have to|has to|must|going to|gonna|plan to|"
    r"let'?s (?:send|prepare|create|schedule|update|fix|implement|review|draft|reach out))\b",
    re.I,
)
_QUESTION = re.compile(
    r"\b(question|open question|need to figure out|not sure|we should discuss|should we|can we|"
    r"how do we|what about|unclear|blocker|risk|concern)\b|\?",
    re.I,
)
_OWNER = re.compile(r"\b([A-Z][a-z]{1,20})\s+(will|should|need to|has to|must|is going to|is planning to)\b")
_DUE = re.compile(
    r"\b(?:by|before)\s+((?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday|tomorrow|today|eod|end of (?:day|week)|next week|this week))\b",
    re.I,
)
# Pronouns should never be treated as action-item owners.
_PRONOUNS = {"We", "I", "You", "They", "He", "She", "It"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _content_words(text: str) -> list[str]:
    return [w for w in _tokenize(text) if len(w) > 2 and w not in _STOPWORDS and w not in _MEETING_WORDS]


def _sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if len(p.strip()) > 3]


def _score(sentences: list[str]) -> dict[str, float]:
    tf: Counter = Counter()
    for s in sentences:
        tf.update(_content_words(s))
    if not tf:
        return {}
    max_freq = max(tf.values())
    scores: dict[str, float] = {}
    for s in sentences:
        tokens = _tokenize(s)
        if not tokens:
            continue
        raw = sum(tf.get(w, 0) for w in _content_words(s))
        if raw <= 0:
            continue
        scores[s] = round((raw / max_freq) / (len(tokens) ** 0.5), 4)
    return scores


def _cluster(segments: list[Segment], gap: float = 90.0) -> list[list[Segment]]:
    clusters: list[list[Segment]] = []
    current: list[Segment] = []
    prev_end: float | None = None
    for seg in segments:
        if prev_end is not None and seg.start - prev_end > gap:
            if current:
                clusters.append(current)
                current = []
        current.append(seg)
        prev_end = seg.end
    if current:
        clusters.append(current)
    return clusters


def _label(text: str) -> str:
    words = [w for w, _ in Counter(_content_words(text)).most_common(3)]
    return " ".join(w.capitalize() for w in words) if words else "Discussion"


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class ExtractiveSummarizer(Summarizer):
    def summarize(self, transcript: Transcript, title: str | None = None) -> MeetingSummary:
        segments = transcript.segments or [Segment(text=transcript.text)]
        sentences = [
            (s, seg.speaker) for seg in segments for s in _sentences(seg.text)
        ]
        raw = [s for s, _ in sentences]
        scored = _score(raw)
        top_sentences = [s for s, _ in sorted(scored.items(), key=lambda kv: -kv[1])]
        executive = " ".join(top_sentences[:3]) or transcript.text.strip()[:300]

        speakers = sorted({seg.speaker for seg in segments if seg.speaker})

        discussion_points: list[DiscussionPoint] = []
        for cluster in _cluster(segments)[:6]:
            cluster_text = " ".join(seg.text for seg in cluster)
            label = _label(cluster_text)
            cluster_sents = _sentences(cluster_text)
            best = sorted(
                ((s, scored.get(s, 0.0)) for s in cluster_sents),
                key=lambda kv: -kv[1],
            )[:2]
            discussion_points.append(DiscussionPoint(topic=label, summary=" ".join(s for s, _ in best)))

        decisions = _dedupe([s for s in raw if _DECISION.search(s)])[:8]

        decision_set = set(decisions)
        action_sentences = _dedupe(
            [s for s in raw if (not decision_set or s not in decision_set) and (_ACTION.search(s) or _FUTURE.search(s))]
        )
        action_items: list[ActionItem] = []
        speaker_by_sentence = dict(sentences)
        for s in action_sentences[:10]:
            owner_match = _OWNER.search(s)
            owner = None
            if owner_match and owner_match.group(1) not in _PRONOUNS:
                owner = owner_match.group(1)
            elif speaker_by_sentence.get(s) not in _PRONOUNS:
                owner = speaker_by_sentence.get(s)
            due_match = _DUE.search(s)
            due = due_match.group(1).strip() if due_match else None
            action_items.append(ActionItem(task=s, owner=owner, due=due))

        open_questions = _dedupe([s for s in raw if _QUESTION.search(s)])[:8]

        return MeetingSummary(
            title=title or "Untitled Meeting",
            attendees=speakers,
            executive_summary=executive,
            key_topics=[p.topic for p in discussion_points],
            discussion_points=discussion_points,
            decisions=decisions,
            action_items=action_items,
            open_questions=open_questions,
            next_steps=[a.task for a in action_items],
        )
