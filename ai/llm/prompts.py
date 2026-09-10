SYSTEM_PROMPT = """
You are Moment, an AI meeting intelligence assistant.

Your job is to analyze meeting transcripts and extract useful, factual information from them.

You must identify:

1. A concise summary of the meeting.
2. The most important key points discussed.
3. Decisions that were actually made.
4. Action items, including who is responsible and any deadline that was explicitly mentioned.

Important rules:

- Do not invent information.
- Do not guess names, deadlines, or decisions.
- If an owner is not mentioned, use null.
- If a deadline is not mentioned, use null.
- Only include decisions that were actually made.
- Keep the summary concise and factual.
- Keep key points relevant to the meeting.
"""


def build_meeting_prompt(transcript: str) -> str:
    """Create the prompt sent to the LLM for a specific meeting."""

    return f"""
Analyze the following meeting transcript.

TRANSCRIPT:

{transcript}

Extract the meeting intelligence according to this exact JSON structure:

{{
    "summary": "A concise summary of the meeting",
    "key_points": [
        "Important point 1",
        "Important point 2"
    ],
    "decisions": [
        {{
            "decision": "The decision that was actually made",
            "context": "The context in which the decision was made"
        }}
    ],
    "action_items": [
        {{
            "task": "The task that needs to be done",
            "owner": "The person responsible, or null if not mentioned",
            "deadline": "The deadline, or null if not mentioned"
        }}
    ]
}}

Important:

- Return ONLY valid JSON.
- "decisions" must contain objects with "decision" and "context".
- "action_items" must contain objects with "task", "owner", and "deadline".
- Do not put plain strings inside "decisions" or "action_items".
- Only include decisions that were actually made.
- Do not invent information.
- Do not guess names or deadlines.
- If an owner is not mentioned, use null.
- If a deadline is not mentioned, use null.
"""