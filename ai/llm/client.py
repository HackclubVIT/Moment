from groq import Groq

from ai.llm.config import GROQ_API_KEY, GROQ_LLM_MODEL
from ai.llm.prompts import SYSTEM_PROMPT, build_meeting_prompt
from ai.llm.schemas import MeetingIntelligence


class GroqLLMClient:
    """Client responsible for communicating with the Groq LLM."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add your API key to the .env file."
            )

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_LLM_MODEL

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

        return MeetingIntelligence.model_validate_json(content)