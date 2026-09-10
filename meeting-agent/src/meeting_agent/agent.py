"""The agentic chat bot.

Two brains behind the same chat interface:
- LLM mode (Anthropic or OpenAI key present): a tool-calling loop. The model
  reasons about the user's request and calls join_meeting / start_recording /
  stop_recording / transcribe / summarize / write_mom.
- Rule mode (no key / ``LLM_ENGINE=offline``): a deterministic planner that
  parses meeting URLs and commands and runs the same pipeline.
"""

from __future__ import annotations

import json
import re

from .config import Settings
from .meeting.detect import find_meeting_url
from .pipeline import MeetingPipeline

TOOLS = [
    {
        "name": "join_meeting",
        "description": "Join a video meeting by URL (Google Meet, Zoom, or Teams) in a real browser window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full meeting URL, e.g. https://meet.google.com/abc-defg-hij"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "start_recording",
        "description": "Begin capturing the meeting's audio from the browser tab (or the microphone if tab capture is unavailable).",
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["tab", "mic"], "default": "tab"}
            },
        },
    },
    {
        "name": "stop_recording",
        "description": "Stop the current recording and save the audio file. Returns the audio file path.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "transcribe",
        "description": "Convert the recorded meeting audio into a text transcript (speech-to-text).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "summarize",
        "description": "Summarize the transcript into structured minutes: decisions, action items, topics, open questions.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "write_mom",
        "description": "Render the summary into a Minutes of Meeting (MOM) markdown document and save it to disk. Returns the file path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Optional meeting title for the document."}
            },
        },
    },
    {
        "name": "status",
        "description": "Report the current state of the session (joined? recording? audio/transcript/summary/MOM paths).",
        "input_schema": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = """You are Meeting Agent, an AI assistant that attends online meetings on the user's behalf and produces Minutes of Meeting (MOM).

You have tools that operate on a single live meeting session:
- join_meeting: open a meeting URL in a real browser and join it.
- start_recording: capture the meeting audio (tab mode captures remote speakers; mic mode captures the user's microphone).
- stop_recording: finish the recording and save the audio file.
- transcribe: convert the audio to text.
- summarize: derive decisions, action items, topics and open questions from the transcript.
- write_mom: render the summary as a Minutes of Meeting markdown document.

Workflow guidance:
1. If the user provides a meeting URL, join it, then start recording, and tell the user recording has begun and they can type 'stop' when the meeting ends. Keep replies brief.
2. When the user says the meeting is over (stop/done/end), stop the recording, transcribe, summarize, and write the MOM.
3. When the MOM is written, reply with its file path and a 2-3 sentence overview.
4. If a tool reports an error, report it plainly and suggest the next sensible step.
5. Do not invent facts; the tools report what actually happened."""

HELP_TEXT = """I can attend a meeting for you and produce the Minutes of Meeting (MOM).

Try things like:
  join https://meet.google.com/xxx and record it
  stop                (when the meeting ends - I transcribe, summarize and write the MOM)
  /status             (current session state)
  /transcribe         (convert the recording to text)
  /summarize          (extract decisions and action items)
  /mom                (write the MOM document)
  /help               (this help)
  /quit

An LLM key (ANTHROPIC_API_KEY or OPENAI_API_KEY) unlocks the full agentic brain;
without one I run a rule-based flow with the same capabilities."""


class Agent:
    def __init__(self, settings: Settings, pipeline: MeetingPipeline):
        self.settings = settings
        self.pipeline = pipeline

    # -- public ---------------------------------------------------------------
    def reply(self, user_text: str) -> str:
        text = (user_text or "").strip()
        if not text:
            return HELP_TEXT
        if text.startswith("/"):
            return self._command(text)
        if self._has_llm():
            return self._llm_loop(text)
        return self._rule_reply(text)

    # -- tool execution -------------------------------------------------------
    def _has_llm(self) -> bool:
        return (
            self.settings.llm_engine != "offline"
            and bool(self.settings.anthropic_api_key or self.settings.openai_api_key)
        )

    def _run_tool(self, name: str, args: dict) -> str:
        p = self.pipeline
        try:
            if name == "join_meeting":
                result = p.join(args["url"])
                return json.dumps({"ok": result["ok"], "message": result["message"]})
            if name == "start_recording":
                return json.dumps({"ok": True, "message": p.start_recording(args.get("mode", "tab"))})
            if name == "stop_recording":
                return json.dumps({"ok": True, "message": p.stop_recording()})
            if name == "transcribe":
                return json.dumps({"ok": True, "message": p.transcribe()})
            if name == "summarize":
                return json.dumps({"ok": True, "message": p.summarize()})
            if name == "write_mom":
                return json.dumps({"ok": True, "message": p.write_mom(args.get("title"))})
            if name == "status":
                return json.dumps({"ok": True, "status": p.status_text()})
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        return json.dumps({"ok": False, "error": f"unknown tool: {name}"})

    # -- LLM tool loop --------------------------------------------------------
    def _llm_loop(self, user_text: str) -> str:
        if self.settings.anthropic_api_key:
            return self._anthropic_loop(user_text)
        return self._openai_loop(user_text)

    def _anthropic_loop(self, user_text: str) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.settings.anthropic_api_key)
        tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in TOOLS
        ]
        messages = [{"role": "user", "content": user_text}]
        for _ in range(self.settings.agent_max_steps):
            resp = client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )
            if resp.stop_reason == "tool_use":
                assistant_blocks = []
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        assistant_blocks.append(
                            {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                        )
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": self._run_tool(block.name, block.input),
                            }
                        )
                    elif block.type == "text" and block.text.strip():
                        assistant_blocks.append({"type": "text", "text": block.text})
                messages.append({"role": "assistant", "content": assistant_blocks})
                messages.append({"role": "user", "content": results})
                continue
            return "".join(b.text for b in resp.content if b.type == "text").strip() or "Done."
        return "Reached the step limit.\n" + self.pipeline.status_text()

    def _openai_loop(self, user_text: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url or None,
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in TOOLS
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        for _ in range(self.settings.agent_max_steps):
            resp = client.chat.completions.create(
                model=self.settings.openai_llm_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []
            if not tool_calls:
                return msg.content or "Done."
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": self._run_tool(tc.function.name, args)}
                )
        return "Reached the step limit.\n" + self.pipeline.status_text()

    # -- rule-based fallback --------------------------------------------------
    def _rule_reply(self, text: str) -> str:
        low = text.lower()
        url = find_meeting_url(text)
        if url:
            lines = []
            try:
                result = self.pipeline.join(url)
                lines.append(result["message"])
                try:
                    lines.append(self.pipeline.start_recording("tab"))
                except Exception as exc:
                    lines.append(f"Could not start recording automatically: {exc}")
                return "Joined the meeting and started recording.\n" + "\n".join(lines)
            except Exception as exc:
                return f"Could not join the meeting: {exc}"

        recording = bool(self.pipeline.recorder and self.pipeline.recorder.status().get("recording"))
        stop_words = ("stop", "done", "end", "over", "finish", "wrap")
        if low.strip() in ("stop", "done", "end", "stop recording", "meeting over", "meeting ended", "we're done") or (
            recording and any(w in low for w in stop_words)
        ):
            if not self.pipeline.audio_path and not recording:
                return "I'm not recording anything right now. Give me a meeting URL to join first."
            try:
                parts = [
                    self.pipeline.stop_recording(),
                    self.pipeline.transcribe(),
                    self.pipeline.summarize(),
                    self.pipeline.write_mom(),
                ]
                return "Meeting wrapped up.\n" + "\n".join(parts) + "\n\n" + self._mom_preview()
            except Exception as exc:
                return f"Error finishing the meeting: {exc}"

        if low.strip() in ("status",):
            return self.pipeline.status_text()
        if "transcribe" in low and self.pipeline.audio_path:
            return self.pipeline.transcribe()
        if "summar" in low and self.pipeline.transcript:
            return self.pipeline.summarize()
        if ("mom" in low or "minutes" in low) and self.pipeline.transcript:
            try:
                return self.pipeline.summarize() + "\n" + self.pipeline.write_mom()
            except Exception as exc:
                return f"Could not build the MOM: {exc}"
        return HELP_TEXT

    def _mom_preview(self) -> str:
        s = self.pipeline.summary
        if not s:
            return ""
        lines = [f"Summary: {s.executive_summary or s.title}"]
        if s.decisions:
            lines.append("Decisions: " + "; ".join(s.decisions[:3]))
        if s.action_items:
            lines.append("Action items: " + "; ".join(a.task[:80] for a in s.action_items[:3]))
        return "\n".join(lines)

    # -- slash commands -------------------------------------------------------
    def _command(self, text: str) -> str:
        parts = text[1:].strip().split(None, 1)
        cmd = (parts[0] or "").lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        p = self.pipeline
        if cmd == "join":
            if not arg:
                return "Usage: /join <meeting-url>"
            result = p.join(arg)
            return result["message"]
        if cmd == "record":
            mode = "mic" if arg.lower() == "mic" else "tab"
            return p.start_recording(mode)
        if cmd == "stop":
            if not p.audio_path and not (p.recorder and p.recorder.status().get("recording")):
                return "Not recording anything right now."
            try:
                return (
                    p.stop_recording()
                    + "\n"
                    + p.transcribe()
                    + "\n"
                    + p.summarize()
                    + "\n"
                    + p.write_mom()
                )
            except Exception as exc:
                return f"Error finishing the meeting: {exc}"
        if cmd == "transcribe":
            if not p.audio_path:
                return "No recording yet. Join a meeting and record first."
            return p.transcribe()
        if cmd == "summarize":
            if not p.transcript:
                return "No transcript yet. Transcribe first."
            return p.summarize()
        if cmd == "mom":
            if not p.transcript:
                return "No transcript yet. Transcribe first."
            try:
                return p.summarize() + "\n" + p.write_mom()
            except Exception as exc:
                return f"Could not build the MOM: {exc}"
        if cmd == "status":
            return p.status_text()
        if cmd in ("help", "?"):
            return HELP_TEXT
        return HELP_TEXT
