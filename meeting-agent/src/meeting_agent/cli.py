"""Command-line interface for Meeting Agent.

Commands:
  meeting-agent chat                 Interactive agentic chat bot
  meeting-agent run --url <url>      One-shot: join, record, transcribe, summarize, MOM
  meeting-agent mom --transcript f   MOM from an existing transcript (offline-safe)
  meeting-agent transcribe --audio f Speech-to-text a local audio file
  meeting-agent demo                 Offline demo: MOM from the sample transcript
  meeting-agent serve                Run the backend API (FastAPI)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .agent import Agent
from .config import Settings
from .pipeline import MeetingPipeline

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _describe_engines(settings: Settings) -> str:
    if settings.llm_engine == "offline":
        llm = "offline (extractive)"
    elif settings.anthropic_api_key:
        llm = f"anthropic ({settings.anthropic_model})"
    elif settings.openai_api_key:
        llm = f"openai ({settings.openai_llm_model})"
    else:
        llm = "offline (extractive)"
    if settings.stt_engine == "local":
        stt = "local (faster-whisper)"
    elif settings.openai_api_key or settings.stt_engine == "api":
        stt = f"api ({settings.openai_stt_model})"
    else:
        stt = "local (faster-whisper)"
    return f"LLM: {llm} | STT: {stt}"


# -- chat ---------------------------------------------------------------------
def _cmd_chat(args) -> int:
    settings = _settings()
    pipeline = MeetingPipeline(settings)
    agent = Agent(settings, pipeline)
    console.print(
        f"[bold magenta]Meeting Agent[/bold magenta] v{__version__} - "
        f"{_describe_engines(settings)}",
    )
    console.print("Type 'help' for commands, 'quit' to exit.\n")
    try:
        if args.url:
            console.print(agent.reply(args.url))
        while True:
            try:
                user = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not user:
                continue
            if user.lower() in ("quit", "exit", "/quit"):
                break
            reply = agent.reply(user)
            console.print(f"[cyan]Agent>[/cyan] {reply}\n")
    finally:
        pipeline.close()
    return 0


# -- run ----------------------------------------------------------------------
def _cmd_run(args) -> int:
    settings = _settings()
    pipeline = MeetingPipeline(settings)
    try:
        console.print(f"Joining {args.url} …")
        result = pipeline.join(args.url)
        console.print(result["message"])
        if not result["ok"]:
            console.print("[yellow]Auto-join did not complete — the browser is open; "
                          "join manually, then recording will still capture the tab audio.[/yellow]")

        console.print(pipeline.start_recording(args.record_mode))
        minutes = args.duration or settings.default_duration_minutes
        console.print(f"Recording for {minutes} minute(s). Press Ctrl+C to stop early.")
        try:
            for i in range(minutes):
                time.sleep(60)
                console.print(f"  {i + 1}/{minutes} minute(s) elapsed")
        except KeyboardInterrupt:
            console.print("\nStopping early.")
        console.print(pipeline.stop_recording())
        console.print(pipeline.transcribe())
        console.print(pipeline.summarize(offline=args.offline))
        console.print(pipeline.write_mom(args.title))
        console.print(f"\n[bold green]Done![/bold green] MOM: {pipeline.mom_path}")
    finally:
        pipeline.close()
    return 0


# -- mom ----------------------------------------------------------------------
def _cmd_mom(args) -> int:
    settings = _settings()
    from .models import MeetingSummary
    from .mom.renderer import render_mom, save_mom
    from .parsing import load_transcript
    from .summarizer.base import build_summarizer

    if args.transcript:
        transcript = load_transcript(args.transcript)
        source = args.transcript
    elif args.audio:
        from .transcriber.base import build_transcriber

        transcript = build_transcriber(settings).transcribe(args.audio)
        out = settings.output_dir / f"{Path(args.audio).stem}_transcript.txt"
        out.write_text(transcript.text, encoding="utf-8")
        source = str(out)
        console.print(f"Transcribed {len(transcript.segments)} segment(s) -> {out}")
    else:
        console.print("[red]Provide --transcript <file> or --audio <file>[/red]")
        return 2

    summarizer = build_summarizer(
        dataclasses.replace(settings, llm_engine="offline") if args.offline else settings
    )
    summary = summarizer.summarize(transcript, title=args.title)
    path = save_mom(summary, args.out, title=args.title)

    console.print(f"Transcript: {source}")
    console.print(f"Attendees: {', '.join(summary.attendees) or 'not captured'}")
    _print_summary_table(summary)
    console.print(f"\n[bold green]MOM saved:[/bold green] {path}\n")
    if args.preview:
        console.print(render_mom(summary))
    return 0


def _print_summary_table(summary) -> None:
    table = Table(title=summary.title)
    table.add_column("Decisions")
    table.add_column("Action items")
    for decision, item in zip(summary.decisions[:4] or [""], summary.action_items[:4] or [""]):
        table.add_row(decision, f"{item.task} (owner: {item.owner or '-'}, due: {item.due or '-'})")
    if summary.decisions or summary.action_items:
        console.print(table)


# -- transcribe ---------------------------------------------------------------
def _cmd_transcribe(args) -> int:
    settings = _settings()
    from .parsing import save_transcript_text
    from .transcriber.base import build_transcriber

    transcript = build_transcriber(settings).transcribe(args.audio)
    stem = Path(args.audio).stem
    txt = save_transcript_text(transcript, settings.output_dir / f"{stem}_transcript.txt")
    console.print(f"Transcribed {len(transcript.segments)} segment(s) from {args.audio}")
    console.print(f"Transcript saved: {txt}")
    console.print("\n--- Transcript ---")
    console.print(transcript.text)
    return 0


# -- demo ---------------------------------------------------------------------
def _cmd_demo(args) -> int:
    settings = _settings()
    sample = PROJECT_ROOT / "examples" / "sample_transcript.txt"
    console.print("[bold]Offline demo:[/bold] sample transcript -> extractive summary -> MOM\n")

    if args.tts:
        console.print("Generating a synthetic meeting audio file (Windows SAPI TTS)…")
        ps1 = PROJECT_ROOT / "examples" / "make_demo_audio.ps1"
        audio = settings.output_dir / "demo_meeting.wav"
        if not audio.exists():
            import subprocess

            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1), "-Out", str(audio)],
                check=False,
            )
        if audio.exists():
            console.print(f"Audio: {audio}")
            txt = settings.output_dir / "demo_meeting_transcript.txt"
            try:
                _cmd_transcribe(argparse.Namespace(audio=str(audio)))
            except Exception as exc:
                console.print(f"[yellow]Transcription skipped ({exc}); using the sample transcript instead.[/yellow]")
            transcript = txt if txt.exists() else sample
            return _cmd_mom(argparse.Namespace(transcript=str(transcript), audio=None, offline=True, title="Demo Meeting", out=str(settings.output_dir), preview=args.preview))

    return _cmd_mom(argparse.Namespace(transcript=str(sample), audio=None, offline=True, title="Demo Meeting", out=str(settings.output_dir), preview=args.preview))


# -- serve ----------------------------------------------------------------------
def _cmd_serve(args) -> int:
    import uvicorn

    console.print(f"Starting Meeting Agent API on http://{args.host}:{args.port}  (docs: /docs)")
    uvicorn.run("meeting_agent.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# -- main ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting-agent",
        description="Agentic AI bot: joins meetings, records audio, transcribes, and writes Minutes of Meeting.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_chat = sub.add_parser("chat", help="Interactive agentic chat bot")
    p_chat.add_argument("--url", help="Meeting URL to join on startup")

    p_run = sub.add_parser("run", help="Join a meeting, record for N minutes, and produce the MOM")
    p_run.add_argument("--url", required=True, help="Meeting URL (Google Meet, Zoom, Teams)")
    p_run.add_argument("--duration", type=int, default=None, help="Recording duration in minutes")
    p_run.add_argument("--record-mode", choices=["tab", "mic"], default="tab", help="tab = meeting audio, mic = microphone")
    p_run.add_argument("--offline", action="store_true", help="Force the offline extractive summarizer")
    p_run.add_argument("--title", default=None, help="Meeting title for the MOM")

    p_mom = sub.add_parser("mom", help="Produce a MOM from an existing transcript or audio file")
    p_mom.add_argument("--transcript", help="Path to a transcript text file")
    p_mom.add_argument("--audio", help="Path to an audio file to transcribe first")
    p_mom.add_argument("--offline", action="store_true", help="Force the offline extractive summarizer")
    p_mom.add_argument("--title", default=None, help="Meeting title for the MOM")
    p_mom.add_argument("--out", default="output", help="Output directory")
    p_mom.add_argument("--preview", action="store_true", help="Print the rendered MOM")

    p_tr = sub.add_parser("transcribe", help="Speech-to-text a local audio file")
    p_tr.add_argument("--audio", required=True, help="Path to an audio file (wav, mp3, webm, m4a, …)")

    p_demo = sub.add_parser("demo", help="Offline demo: MOM from the sample transcript")
    p_demo.add_argument("--tts", action="store_true", help="Also synthesize a demo audio file with Windows TTS and transcribe it")
    p_demo.add_argument("--preview", action="store_true", help="Print the rendered MOM")

    p_serve = sub.add_parser("serve", help="Run the backend API (FastAPI)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "cmd", None):
        build_parser().print_help()
        return 2
    if args.cmd == "chat":
        return _cmd_chat(args)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "mom":
        return _cmd_mom(args)
    if args.cmd == "transcribe":
        return _cmd_transcribe(args)
    if args.cmd == "demo":
        return _cmd_demo(args)
    if args.cmd == "serve":
        return _cmd_serve(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
