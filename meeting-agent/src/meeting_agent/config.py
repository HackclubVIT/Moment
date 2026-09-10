"""Runtime configuration, read from environment variables and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


@dataclass
class Settings:
    # LLM / summarization
    anthropic_api_key: str | None = _env("ANTHROPIC_API_KEY")
    anthropic_model: str = _env("ANTHROPIC_MODEL", "claude-sonnet-4-5") or "claude-sonnet-4-5"
    openai_api_key: str | None = _env("OPENAI_API_KEY")
    openai_base_url: str | None = _env("OPENAI_BASE_URL")
    openai_llm_model: str = _env("OPENAI_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    llm_engine: str = _env("LLM_ENGINE", "auto") or "auto"  # auto|anthropic|openai|offline

    # Speech-to-text
    stt_engine: str = _env("STT_ENGINE", "auto") or "auto"  # auto|api|local
    openai_stt_model: str = _env("OPENAI_STT_MODEL", "whisper-1") or "whisper-1"
    local_stt_model: str = _env("LOCAL_STT_MODEL", "base") or "base"

    # Output
    output_dir: Path = Path(_env("OUTPUT_DIR", "output") or "output")

    # Backend API / database
    database_url: str = _env("DATABASE_URL", "sqlite:///./meeting_agent.db") or "sqlite:///./meeting_agent.db"

    # Browser / meeting joining
    headless: bool = _bool("HEADLESS", False)
    chrome_user_data_dir: Path = Path(
        _env("CHROME_USER_DATA_DIR", str(Path.home() / ".meeting_agent" / "chrome_profile"))
        or str(Path.home() / ".meeting_agent" / "chrome_profile")
    )
    default_duration_minutes: int = _int("DEFAULT_DURATION_MINUTES", 30)

    # Agent
    agent_max_steps: int = _int("AGENT_MAX_STEPS", 15)
