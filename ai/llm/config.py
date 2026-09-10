from dotenv import load_dotenv
import os

# Load variables from the .env file
load_dotenv()

# Read API key and model name
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL = os.getenv(
    "GROQ_LLM_MODEL",
    "openai/gpt-oss-20b"
)