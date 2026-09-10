"""The end-to-end meeting pipeline: join -> record -> transcribe -> summarize -> MOM."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .config import Settings
from .models import MeetingSummary, Transcript


def _offline_settings(settings: Settings) -> Settings:
    return dataclasses.replace(settings, llm_engine="offline")


class MeetingPipeline:
    """Holds the state of one meeting session and exposes each stage.

    Every stage returns a short human-readable string, which makes the object
    directly usable as the backend for the agent's tool calls.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.browser = None
        self.page = None
        self.recorder = None
        self.platform: str | None = None
        self.audio_path: Path | None = None
        self.transcript: Transcript | None = None
        self.summary: MeetingSummary | None = None
        self.mom_path: Path | None = None

    # -- stage 1: join --------------------------------------------------------
    def _ensure_browser(self) -> None:
        if self.browser is None:
            from .meeting.browser import BrowserSession

            self.browser = BrowserSession(self.settings)
            self.browser.start()

    def join(self, url: str) -> dict:
        from .meeting.detect import detect_platform
        from .meeting.joiners import join_meeting

        self._ensure_browser()
        if self.page is None:
            self.page = self.browser.new_page()
        self.platform = detect_platform(url)
        result = join_meeting(self.page, url, self.platform)

        from .recorder.web_recorder import WebRecorder

        self.recorder = WebRecorder(self.page, self.settings.output_dir)
        self.recorder.attach()
        return result

    # -- stage 2: record ------------------------------------------------------
    def start_recording(self, mode: str = "tab") -> str:
        if self.recorder is None:
            raise RuntimeError("Join a meeting first (no recorder attached).")
        if mode == "mic":
            try:
                self.page.context.grant_permissions(["microphone"], origin=self.page.url)
            except Exception:
                pass
        ok, message = self.recorder.start(mode)
        if not ok:
            raise RuntimeError(message)
        return message

    def stop_recording(self) -> str:
        if self.recorder is None:
            raise RuntimeError("No recorder attached.")
        self.audio_path = self.recorder.stop()
        return f"Recording saved to {self.audio_path}"

    # -- stage 3: transcribe --------------------------------------------------
    def transcribe(self) -> str:
        if self.audio_path is None:
            raise RuntimeError("No recording to transcribe yet.")
        from .transcriber.base import build_transcriber

        transcript = build_transcriber(self.settings).transcribe(self.audio_path)
        self.transcript = transcript
        out = self.settings.output_dir / f"{self.audio_path.stem}_transcript.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(transcript.text, encoding="utf-8")
        return f"Transcribed {len(transcript.segments)} segment(s) -> {out}"

    # -- stage 4: summarize ---------------------------------------------------
    def summarize(self, offline: bool = False) -> str:
        if self.transcript is None:
            raise RuntimeError("Transcribe the recording first.")
        from .summarizer.base import build_summarizer

        settings = _offline_settings(self.settings) if offline else self.settings
        summary = build_summarizer(settings).summarize(self.transcript)
        self.summary = summary
        return (
            f"Summary ready: {len(summary.decisions)} decision(s), "
            f"{len(summary.action_items)} action item(s), {len(summary.key_topics)} topic(s)."
        )

    # -- stage 5: MOM ---------------------------------------------------------
    def write_mom(self, title: str | None = None) -> str:
        if self.summary is None:
            raise RuntimeError("Summarize the transcript first.")
        from .mom.renderer import save_mom

        self.mom_path = save_mom(self.summary, self.settings.output_dir, title=title)
        return f"MOM saved to {self.mom_path}"

    # -- helpers --------------------------------------------------------------
    def status_text(self) -> str:
        parts = [f"Platform: {self.platform or 'n/a'}"]
        recording = bool(self.recorder and self.recorder.status().get("recording"))
        parts.append(f"Recording: {'active' if recording else 'inactive'}")
        if self.audio_path:
            parts.append(f"Audio: {self.audio_path}")
        if self.transcript:
            parts.append(f"Transcript: {len(self.transcript.segments)} segment(s)")
        if self.summary:
            parts.append(f"Summary: {len(self.summary.action_items)} action item(s)")
        if self.mom_path:
            parts.append(f"MOM: {self.mom_path}")
        return "\n".join(parts)

    def close(self) -> None:
        if self.browser is not None:
            self.browser.close()
