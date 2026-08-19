# Moment - Meeting Organization, Memory & Extraction Network Technology

**Moment** is an AI-powered meeting note taker that captures meeting
audio, converts it into a transcript using Whisper, generates structured
meeting intelligence using an LLM, and stores meeting knowledge for
RAG-powered **"Ask Anything"** queries.

## Project Structure

``` text
Moment/
├── ai/
│   ├── llm/          # Arshia
│   ├── rag/          # Aman
│   └── whisper/      # Rushaan
├── backend/          # Raghavraj
├── frontend/         # Manan
├── data/
└── docs/
```

## Team Workflow

**Do not push directly to `main`.**

### 1. Get the latest code

``` bash
git checkout main
git pull origin main
```

### 2. Create your own branch

``` bash
git checkout -b feature/your-feature-name
```

Examples:

``` bash
git checkout -b feature/whisper
git checkout -b feature/rag
git checkout -b feature/llm
git checkout -b feature/backend-api
```

### 3. Make your changes

Work primarily inside your assigned folder.

### 4. Commit

``` bash
git add .
git commit -m "feat: describe your change"
```

### 5. Push your branch

``` bash
git push -u origin feature/your-feature-name
```

### 6. Create a Pull Request

Go to GitHub and create a **Pull Request** from your branch into `main`.

``` text
Your Branch
    ↓
Pull Request
    ↓
Review
    ↓
Merge
    ↓
main
```

## Important

-   Never push directly to `main`.
-   Never commit `.env` or API keys.
-   Pull from `main` before starting new work.
-   Keep commits meaningful.
-   If you modify another person's module, coordinate with them first.
-   Test your code before creating a PR.

------------------------------------------------------------------------

**Build. Capture. Remember. Ask Anything.**
