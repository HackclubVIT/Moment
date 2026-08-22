"""CLI entry point for the WhisperX module.

Usage::

    python -m ai.whisper devices
    python -m ai.whisper record --output data/recordings/meeting_123.wav
    python -m ai.whisper transcribe data/recordings/meeting_123.wav --meeting-id meeting_123
    python -m ai.whisper health
    python -m ai.whisper pipeline --meeting-id meeting_123
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

from .config import load_config
from .recorder import AudioRecorder
from .transcriber import transcribe_audio

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _progress(stage: str, progress: float) -> None:
    bar_len = 30
    filled = int(bar_len * progress)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {stage:<15} [{bar}] {progress:.0%}", end="", flush=True)
    if progress >= 1.0:
        print()


def cmd_devices(_args: argparse.Namespace) -> None:
    """List available audio devices."""
    devices = AudioRecorder.list_devices()
    print(f"\n{'Idx':<5} {'Name':<45} {'In':<4} {'Out':<4} {'Rate':<8} {'Host API'}")
    print("─" * 90)
    for d in devices:
        print(
            f"{d['index']:<5} {d['name']:<45} {d['max_input_channels']:<4} "
            f"{d['max_output_channels']:<4} {d['default_samplerate']:<8.0f} {d['hostapi']}"
        )

    loopback = AudioRecorder.find_loopback_device()
    if loopback is not None:
        print(f"\n  ✓ Auto-detected loopback device: [{loopback}]")
    else:
        print("\n  ✗ No loopback device detected (system audio capture unavailable)")


def cmd_record(args: argparse.Namespace) -> None:
    """Record audio from mic (+ system audio if available)."""
    rec = AudioRecorder()

    system_device = None if args.mic_only else "auto"
    rec.start(system_device=system_device)
    print("🎤 Recording… press Ctrl+C to stop\n")

    try:
        while rec.is_recording:
            elapsed = rec.elapsed
            mins, secs = divmod(int(elapsed), 60)
            print(f"\r  ⏱  {mins:02d}:{secs:02d}", end="", flush=True)
            import time
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    output_dir = args.output if args.output else None
    path = rec.stop(output_dir=output_dir)
    print(f"\n\n✓ Saved: {path}")


def cmd_transcribe(args: argparse.Namespace) -> None:
    """Transcribe an existing audio file."""
    meeting_id = args.meeting_id or f"meeting_{uuid.uuid4().hex[:8]}"
    print(f"\n🔊 Transcribing: {args.audio_path}")
    print(f"   Meeting ID:   {meeting_id}\n")

    result = transcribe_audio(
        audio_path=args.audio_path,
        meeting_id=meeting_id,
        language=args.language,
        translate=not args.no_translate,
        on_progress=_progress,
    )

    if result["status"] == "error":
        print(f"\n✗ Error: {result['error']}")
        sys.exit(1)

    meta = result["metadata"]
    print(f"\n✓ Done in {meta['processing_time_seconds']:.1f}s")
    print(f"  Segments:  {meta['num_segments']}")
    print(f"  Speakers:  {meta['num_speakers']}")
    print(f"  Language:  {meta['language_detected']}")
    print(f"  Mode:      {meta['inference_mode']}")
    print(f"  Saved:     {meta['transcript_path']}")


def cmd_health(_args: argparse.Namespace) -> None:
    """Check Groq API connectivity and configuration."""
    cfg = load_config()
    if not cfg.groq_api_key:
        print("✗ GROQ_API_KEY not set in .env")
        sys.exit(1)

    try:
        from .groq_whisper import GroqWhisper

        client = GroqWhisper(config=cfg)
        status = client.check_health()
        if status.get("status") == "ok":
            print(f"✓ Groq API connected successfully!")
            print(f"  Configured model: {status.get('configured_model')}")
            print(f"  Available models: {', '.join(status.get('available_whisper_models', []))}")
        else:
            print(f"✗ Groq API check failed: {status.get('message')}")
            sys.exit(1)
    except Exception as exc:
        print(f"✗ Groq health check failed: {exc}")
        sys.exit(1)


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Full pipeline: record → transcribe."""
    meeting_id = args.meeting_id or f"meeting_{uuid.uuid4().hex[:8]}"

    # Record
    rec = AudioRecorder()
    rec.start(system_device=None if args.mic_only else "auto")
    print(f"🎤 Recording for meeting '{meeting_id}'… press Ctrl+C to stop\n")

    try:
        while rec.is_recording:
            elapsed = rec.elapsed
            mins, secs = divmod(int(elapsed), 60)
            print(f"\r  ⏱  {mins:02d}:{secs:02d}", end="", flush=True)
            import time
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    audio_path = rec.stop()
    print(f"\n\n✓ Recording saved: {audio_path}")
    print(f"\n🔊 Starting transcription…\n")

    # Transcribe
    result = transcribe_audio(
        audio_path=audio_path,
        meeting_id=meeting_id,
        language=args.language,
        translate=not args.no_translate,
        on_progress=_progress,
    )

    if result["status"] == "error":
        print(f"\n✗ Error: {result['error']}")
        sys.exit(1)

    meta = result["metadata"]
    print(f"\n✓ Pipeline complete!")
    print(f"  Audio:     {meta['audio_path']}")
    print(f"  Transcript:{meta['transcript_path']}")
    print(f"  Segments:  {meta['num_segments']}")
    print(f"  Speakers:  {meta['num_speakers']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ai.whisper",
        description="Moment AI — Audio capture & Whisper transcription",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # devices
    sub.add_parser("devices", help="List available audio devices")

    # record
    p_rec = sub.add_parser("record", help="Record audio")
    p_rec.add_argument("--output", "-o", help="Output directory for the WAV file")
    p_rec.add_argument("--mic-only", action="store_true", help="Skip system audio capture")

    # transcribe
    p_tr = sub.add_parser("transcribe", help="Transcribe an audio file")
    p_tr.add_argument("audio_path", help="Path to audio file")
    p_tr.add_argument("--meeting-id", "-m", help="Meeting identifier")
    p_tr.add_argument("--language", "-l", default="auto", help="Language code or 'auto'")
    p_tr.add_argument("--no-translate", action="store_true", help="Skip English translation")

    # health
    sub.add_parser("health", help="Check Groq API connectivity and models")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Record then transcribe")
    p_pipe.add_argument("--meeting-id", "-m", help="Meeting identifier")
    p_pipe.add_argument("--language", "-l", default="auto", help="Language code or 'auto'")
    p_pipe.add_argument("--no-translate", action="store_true", help="Skip English translation")
    p_pipe.add_argument("--mic-only", action="store_true", help="Skip system audio capture")

    args = parser.parse_args()

    commands = {
        "devices": cmd_devices,
        "record": cmd_record,
        "transcribe": cmd_transcribe,
        "health": cmd_health,
        "pipeline": cmd_pipeline,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
