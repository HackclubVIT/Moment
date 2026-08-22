# Getting Started with the Whisper Module

This guide explains how to set up, run, test, and integrate the **Audio Capture & Whisper Transcription** module into the **Moment** application.

---

## 1. System Prerequisites

The module handles low-level audio capture and processing. Install the required system tools:

### macOS
```bash
# Required for audio recording and audio format processing
brew install portaudio ffmpeg

# Optional (Recommended): Virtual audio loopback driver to capture meeting audio
# (Zoom, Google Meet, Teams, Webex) in addition to your microphone
brew install --cask blackhole-2ch
```

> **Setting up System Audio Capture (Optional):**
> Open **System Settings → Sound**. To capture meeting sound while still hearing it through your speakers/headphones, create a **Multi-Output Device** in macOS *Audio MIDI Setup* combining your physical output (headphones/speakers) and *BlackHole 2ch*.

---

## 2. Python Environment Setup

Create and activate a virtual environment, then install the Python dependencies:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install dependencies
pip install -r ai/whisper/requirements.txt
```

---

## 3. Environment Variables Configuration

Copy `.env.example` to `.env` in the root of the project:

```bash
cp .env.example .env
```

Open `.env` and set your credentials:

```env
# ── Groq Cloud (Speech-to-Text API) ──────────────────────────────────────────
# Get a free API key at: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_WHISPER_MODEL=whisper-large-v3

# ── HuggingFace (Optional for local speaker diarization) ─────────────────────
# Only needed if you want offline speaker diarization with pyannote
HF_TOKEN=hf_your_token_here
```

---

## 4. Verification & Testing

### Test 1: Health Check
Confirm your Groq API key and cloud models are accessible:
```bash
python -m ai.whisper health
```
Expected output:
```text
✓ Groq API connected successfully!
  Configured model: whisper-large-v3
  Available models: whisper-large-v3-turbo, whisper-large-v3
```

### Test 2: Audio Devices Check
List all input and loopback audio devices on your machine:
```bash
python -m ai.whisper devices
```

### Test 3: Run the Full Test Suite
Run all 44 automated unit tests:
```bash
pytest ai/whisper/tests/ -v
```

---

## 5. Usage & CLI Commands

You can run the module directly from the command line:

### Record Audio Only
```bash
# Record from microphone (+ system audio if loopback exists). Press Ctrl+C to stop.
python -m ai.whisper record --output data/recordings/
```

### Transcribe an Existing Audio File
```bash
python -m ai.whisper transcribe data/recordings/meeting.wav --meeting-id meeting_123
```

### Run the Full Pipeline (Live Record → Transcribe)
```bash
python -m ai.whisper pipeline --meeting-id test_meeting_1
```
1. Speak into your microphone.
2. Press `Ctrl + C` to stop.
3. The transcript is immediately generated and saved to `data/transcripts/test_meeting_1.json`.

---

## 6. Python API (For Backend, LLM & RAG Integration)

Team members integrating Whisper into other modules (e.g. backend API, LLM meeting notes, RAG search) can use the single high-level function:

```python
from ai.whisper import transcribe_audio, AudioRecorder

# Option A: Transcribe an existing audio file
result = transcribe_audio(
    audio_path="data/recordings/meeting_123.wav",
    meeting_id="meeting_123",
    language="auto",      # or language code like "en", "es"
    translate=True        # translate non-English audio to English
)

# Option B: Programmatically record audio
rec = AudioRecorder()
rec.start()
# ... meeting in progress ...
audio_path = rec.stop()
result = transcribe_audio(audio_path=audio_path, meeting_id="meeting_123")
```

### Output JSON Schema

`transcribe_audio()` never raises unhandled exceptions. It always returns a dictionary adhering to this schema:

```json
{
  "meeting_id": "meeting_123",
  "status": "completed",
  "error": null,
  "metadata": {
    "audio_duration_seconds": 8.48,
    "processing_time_seconds": 1.24,
    "inference_mode": "groq",
    "model": "whisper-large-v3",
    "language_detected": "en",
    "num_speakers": 1,
    "num_segments": 1,
    "audio_path": "/path/to/data/recordings/meeting_123.wav",
    "transcript_path": "/path/to/data/transcripts/meeting_123.json",
    "created_at": "2026-08-22T05:35:19Z"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 8.48,
      "speaker": "SPEAKER_00",
      "text": "Welcome everyone to the meeting.",
      "words": [
        {
          "word": "Welcome",
          "start": 0.0,
          "end": 0.45
        }
      ]
    }
  ]
}
```

On failure, `"status": "error"`, `"error"` contains the message, and `"segments": []`.
