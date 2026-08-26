from ai.llm.client import GroqLLMClient
from ai.llm.schemas import MeetingIntelligence


def process_transcript(transcript: str) -> MeetingIntelligence:
    """
    Analyze a meeting transcript and return structured meeting intelligence.
    """

    client = GroqLLMClient()
    return client.analyze_transcript(transcript)