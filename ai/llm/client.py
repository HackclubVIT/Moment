import os

from groq import Groq

from ai.llm.config import GROQ_API_KEY, GROQ_LLM_MODEL
from ai.llm.prompts import SYSTEM_PROMPT, build_meeting_prompt
from ai.llm.schemas import MeetingIntelligence


class GroqLLMClient:
    """Client responsible for communicating with the Groq LLM."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add your API key to the .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_LLM_MODEL") or GROQ_LLM_MODEL

    def analyze_transcript(self, transcript: str) -> MeetingIntelligence:
        """Analyze a meeting transcript and return structured intelligence."""

        prompt = build_meeting_prompt(transcript)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_object"
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq LLM returned empty response content; cannot parse meeting intelligence JSON.")

        return MeetingIntelligence.model_validate_json(content)