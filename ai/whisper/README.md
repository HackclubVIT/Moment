# ai/whisper — Audio Capture & Groq Whisper Transcription

Audio capture → Groq Whisper cloud transcription (with local fallback) → speaker diarization pipeline for the Moment AI Note Taker.

## Architecture

```
Microphone ──┐
              ├──► Audio Mixer ──► WAV (16kHz mono) ──► Groq Cloud API (Whisper Large V3) ──► Transcript JSON
System Audio ─┘                                            │
                                                     (on failure / offline)
                                                           ▼
                                                    Local WhisperX
```

## Setup

### 1. System dependencies (macOS)

```bash
brew install portaudio ffmpeg
```

**System audio capture on macOS** requires a virtual audio driver:

```bash
brew install --cask blackhole-2ch
```

Then in System Settings → Sound, set BlackHole as your output device (or create a Multi-Output Device to hear audio and capture simultaneously).

### 2. Python dependencies

```bash
pip install -r ai/whisper/requirements.txt
```

### 3. Environment variables

```bash
cp .env.example .env
# Fill in your tokens in .env
```

Required tokens:

| Variable | Purpose | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | Groq Cloud Whisper API (fast cloud transcription) | [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_WHISPER_MODEL` | Groq Whisper model (`whisper-large-v3` or `whisper-large-v3-turbo`) | Default: `whisper-large-v3` |
| `HF_TOKEN` | Pyannote diarization for local pipeline (gated model) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

> **Note:** For local speaker diarization, you must accept the pyannote model license on HuggingFace before the token will work:
> [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Usage

### Python API (for team integration)

```python
from ai.whisper import transcribe_audio, AudioRecorder

# Record audio
rec = AudioRecorder()
rec.start()
# ... meeting happens ...
audio_path = rec.stop()

# Transcribe
result = transcribe_audio(
    audio_path=audio_path,
    meeting_id="meeting_123",
)

# result["segments"] → list of {id, start, end, speaker, text, words}
# result["status"]   → "completed" or "error"
```

### CLI

```bash
# List audio devices
python -m ai.whisper devices

# Record audio
python -m ai.whisper record --output data/recordings/

# Transcribe a file
python -m ai.whisper transcribe data/recordings/meeting.wav --meeting-id meeting_123

# Full pipeline (record + transcribe)
python -m ai.whisper pipeline --meeting-id meeting_123

# Health check Groq API connection
python -m ai.whisper health
```

## Output JSON Schema

```json
{
  "meeting_id": "meeting_123",
  "status": "completed",
  "error": null,
  "metadata": {
    "audio_duration_seconds": 3600.5,
    "processing_time_seconds": 12.4,
    "inference_mode": "groq",
    "model": "whisper-large-v3",
    "language_detected": "en",
    "num_speakers": 3,
    "num_segments": 247,
    "audio_path": "data/recordings/meeting_123.wav",
    "transcript_path": "data/transcripts/meeting_123.json",
    "created_at": "2026-08-20T10:30:00Z"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "speaker": "SPEAKER_00",
      "text": "Welcome everyone to the meeting.",
      "words": [
        { "word": "Welcome", "start": 0.0, "end": 0.45 }
      ]
    }
  ]
}
```

On error, `status` is `"error"`, `error` contains the message, and `segments` is `[]`.

## Testing

```bash
pytest ai/whisper/tests/ -v
```

## File Structure

```
ai/whisper/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── config.py            # Configuration from .env
├── recorder.py          # Audio capture (mic + system)
├── transcriber.py       # Main orchestrator
├── groq_whisper.py      # Groq Cloud Whisper API client
├── local_whisper.py     # Local CPU/GPU fallback
├── cleaner.py           # Minimal transcript cleaning
├── requirements.txt     # Python dependencies
├── README.md            # Architecture & API reference
├── GETTING_STARTED.md   # Step-by-step setup and team guide
└── tests/
    ├── conftest.py          # Shared fixtures
    ├── test_config.py
    ├── test_recorder.py
    ├── test_cleaner.py
    ├── test_transcriber.py
    ├── test_schema.py
    ├── test_groq_whisper.py
    └── test_local_whisper.py
```
