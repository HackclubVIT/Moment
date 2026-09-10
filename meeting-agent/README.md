# Meeting Agent

An **agentic AI chat bot** that attends meetings for you and produces the **Minutes of Meeting (MOM)**:

```
you:  "Join https://meet.google.com/xxx and take the minutes"
bot:  🎙 joined -> recording the meeting audio…
      (meeting happens)
you:  "stop"
bot:  -> transcribed (Whisper) -> summarized (LLM) -> MOM.md written
```

It is a single pipeline with pluggable backends, driven by either an LLM
tool-calling agent or a rule-based planner:

```
   join_meeting(url)  -> Playwright opens Chromium and joins Meet / Zoom / Teams
   start_recording()  -> in-page MediaRecorder captures the meeting's tab audio (or mic)
   stop_recording()   -> WebM/Opus file saved to output/
   transcribe()       -> Whisper (hosted API or local faster-whisper) -> timestamped transcript
   summarize()        -> LLM (Anthropic/OpenAI) or offline extractive summarizer
   write_mom()        -> Minutes of Meeting rendered as Markdown
```

---

## Quick start

```bash
cd meeting-agent

# 1) Install core dependencies (agent + summarizer + CLI)
uv sync

# 2) Meeting joining requires Playwright (browser automation):
uv sync --extra meeting
uv run playwright install chromium

# 3) Optional: fully offline speech-to-text
uv sync --extra local-stt

# 4) Add an API key (for the agentic brain + smarter summaries)
cp .env.example .env     # then fill in ANTHROPIC_API_KEY (recommended) or OPENAI_API_KEY
```

> Everything works with **no API keys at all** — the bot falls back to a
> rule-based planner, an offline extractive summarizer, and local Whisper.

## Try it instantly (no meeting needed)

```bash
uv run python -m meeting_agent demo                 # sample transcript -> MOM (offline)
uv run python -m meeting_agent demo --tts           # also synthesize audio, transcribe it (Windows)
uv run python -m meeting_agent mom --transcript examples/sample_transcript.txt --preview
```

## Usage

### Chat bot (agentic)

```bash
uv run python -m meeting_agent chat
```

```text
You> join https://meet.google.com/abc-defg-hij and take minutes
Agent> Joined the meeting and started recording. …
You> stop
Agent> Meeting wrapped up.
       Recording saved to output/recording_20260816_143010.webm
       Transcribed 214 segment(s) -> output/…_transcript.txt
       Summary ready: 3 decision(s), 5 action item(s).
       MOM saved to output/MOM_20260816_143012_untitled-meeting.md
```

With an LLM key the bot uses **tool calling** (it reasons, calls
`join_meeting`, `start_recording`, `stop_recording`, `transcribe`,
`summarize`, `write_mom`, and reports back). Without a key it runs the same
pipeline through a deterministic planner. Slash commands (`/join`, `/record`,
`/stop`, `/transcribe`, `/summarize`, `/mom`, `/status`, `/help`) always work.

### One-shot run (join + record N minutes + MOM)

```bash
uv run python -m meeting_agent run --url https://meet.google.com/abc-defg-hij --duration 45
```

### From an existing transcript or audio file

```bash
uv run python -m meeting_agent mom --transcript meeting.txt
uv run python -m meeting_agent mom --audio recording.webm
uv run python -m meeting_agent transcribe --audio recording.webm
```

## How it records the meeting

When the bot joins a meeting it injects a small `MediaRecorder` into the tab:

- **`tab` mode (default)** — calls `getDisplayMedia({ audio: true,
  preferCurrentTab: true })`. Because Chromium is launched with
  `--auto-select-tab-capture-source`, the *tab's* audio (all remote
  participants) is captured automatically with no picker dialog on
  Chrome ≥ 116. Output is WebM/Opus — **no ffmpeg required**.
- **`mic` mode** — falls back to `getUserMedia` (your microphone).

The bot uses a **persistent Chrome profile** (`~/.meeting_agent/chrome_profile`)
so your Google/Zoom/Teams login persists between runs. Run headed
(`HEADLESS=0`) the first time and log in once.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Agent brain + summarizer (recommended) |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Anthropic model |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | – | OpenAI/OpenAI-compatible LLM + STT |
| `LLM_ENGINE` | `auto` | `auto` \| `anthropic` \| `openai` \| `offline` |
| `STT_ENGINE` | `auto` | `auto` \| `api` \| `local` |
| `LOCAL_STT_MODEL` | `base` | faster-whisper model size |
| `OUTPUT_DIR` | `output` | recordings, transcripts, MOMs, uploaded audio |
| `DATABASE_URL` | `sqlite:///./meeting_agent.db` | backend API database (swap for Postgres in deployment) |
| `HEADLESS` | `0` | keep headed so you can see/log in |
| `CHROME_USER_DATA_DIR` | `~/.meeting_agent/chrome_profile` | persistent browser profile |
| `DEFAULT_DURATION_MINUTES` | `30` | length for `run` |

## Project layout

```
src/meeting_agent/
├── cli.py             # chat / run / mom / transcribe / demo / serve commands
├── agent.py           # agentic chat bot (LLM tool loop + rule fallback)
├── pipeline.py        # in-process join -> record -> transcribe -> summarize -> MOM (CLI path)
├── app.py             # FastAPI backend: the persisted meeting API
├── service.py         # meeting lifecycle orchestration + error handling for the API
├── db.py / dbmodels.py# SQLAlchemy engine/session + schema (Meeting, TranscriptSegment, Summary, ...)
├── schemas.py         # API request/response models
├── storage.py         # audio storage abstraction (LocalStorage today)
├── meeting/          # Playwright browser session + Meet/Zoom/Teams joiners
├── recorder/         # in-page MediaRecorder (tab audio or mic)
├── transcriber/      # OpenAI-compatible STT + local faster-whisper
├── summarizer/       # Anthropic/OpenAI LLM + offline extractive
├── mom/              # Markdown MOM renderer
├── models.py         # Transcript / MeetingSummary schemas (in-process pipeline)
└── parsing.py        # transcript file parsing & serialization
tests/                # pytest suite (offline, no network) incl. API tests
examples/             # sample transcript + Windows TTS demo script
```

## Backend API

Alongside the CLI/chat bot there's a persisted backend: a FastAPI service backed
by a SQLite database (swap `DATABASE_URL` for Postgres later — nothing else
changes), so a meeting's transcript, summary, decisions, action items and
lifecycle status survive past a single CLI run and can be consumed by a
frontend or other services.

```bash
uv run python -m meeting_agent serve          # http://127.0.0.1:8000, docs at /docs
uv run python -m meeting_agent serve --reload # development, auto-reload
```

### Endpoints

```
POST /meetings                          create a meeting
GET  /meetings                          list meetings
GET  /meetings/{id}                     status + metadata

POST /meetings/{id}/audio               upload a recorded audio file
POST /meetings/{id}/transcribe          run speech-to-text, persist segments
POST /meetings/{id}/process             run summarization, persist summary/decisions/action items/topics

GET  /meetings/{id}/transcript          raw transcript segments (kept even after summarizing)
GET  /meetings/{id}/summary             executive summary, key points, decisions, action items, topics, open questions
GET  /meetings/{id}/action-items        just the action items

POST /ask                               naive keyword search over stored transcripts (see below)
```

### Meeting lifecycle

Each meeting's `status` is persisted in the database and moves through:

```
created -> uploaded -> transcribing -> transcribed -> processing -> indexed -> completed
                                    \-> failed (failed_stage + error_message set)
```

A failed stage doesn't need a separate retry endpoint — fix the underlying
problem (e.g. the STT backend) and POST the same stage endpoint again; each
stage overwrites its own rows, so retries are naturally idempotent.

### Database schema

`Meeting` is the hub; `Participant`, `TranscriptSegment`, `Summary`,
`ActionItem`, `Decision` and `Topic` all hang off it by `meeting_id`. Raw
transcript segments are kept permanently (not just the generated summary) so
they remain available as evidence for `/ask`. There's no `Users`/auth table in
this MVP — add one behind an `owner_id` FK on `Meeting` if/when auth becomes a
real requirement.

### Module integration points

- **Transcription** (`src/meeting_agent/transcriber/`): today this calls
  Whisper directly (hosted API or local faster-whisper). This is the seam
  where Rushaan's transcription service would plug in instead — swap what
  `build_transcriber()` returns; `service.run_transcription()` doesn't change.
- **Meeting intelligence** (`src/meeting_agent/summarizer/`): today this calls
  an LLM directly (Anthropic/OpenAI, or an offline extractive fallback). This
  is the seam for Arshia's module — swap what `build_summarizer()` returns.
- **RAG / `/ask`** (`src/meeting_agent/service.py::ask`): currently a naive
  keyword search over stored transcript segments, not real retrieval. It's a
  placeholder standing in for Aman's RAG module — the request/response shape
  already matches the agreed contract, so wiring in real vector search means
  replacing this function's body, not the API.

### Storage

Uploaded audio is written to `OUTPUT_DIR/audio/{meeting_id}/` through a
`Storage` interface (`src/meeting_agent/storage.py`); `LocalStorage` is the
only implementation today. Moving to S3/object storage for deployment means
adding a new `Storage` implementation, not touching any caller.

## Notes & limitations

- **Joining real meetings** is best-effort: Google Meet auto-join is the most
  reliable; Zoom/Teams web flows change often and may need a manual click in
  the opened browser (recording still works once you're in).
- **Attendee names** come from speaker labels in the transcript; basic Whisper
  output has no diarization, so `attendees` will be empty unless the STT
  backend provides speakers.
- **Auto tab capture** needs Chrome/Chromium ≥ 116; otherwise the browser shows
  a "share this tab" picker that you accept once.
- **Ethics**: only join meetings you're invited to, follow the host's
  recording policy, and tell participants the bot is recording.

## Tests

```bash
uv run pytest -q
```
