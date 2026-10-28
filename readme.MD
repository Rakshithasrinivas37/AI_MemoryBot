# MemoryBot

An AI chatbot that remembers previous conversations, searches the web for current information, and uses saved memories to provide contextual answers.

## Features

- **Persistent memory:** saves a title, structured summary, and original conversation in ChromaDB.
- **Memory viewer:** search saved memories, click to view the conversation and summary, or delete them.
- **Hybrid retrieval:** combines vector search, BM25, and cross-encoder reranking.
- **Web search:** uses today's date for latest-event queries.
- **Response evaluation:** evaluates answers using conversation context, memories, and tool results.
- **Responsive UI:** supports formatted tables, lists, and code blocks.

## Setup

You need Python 3.10+, a [Groq API key](https://console.groq.com/), and a [Hugging Face token](https://huggingface.co/settings/tokens).

```bash
git clone https://github.com/Rakshithasrinivas37/AI_MemoryBot.git
cd AI_MemoryBot

python3 -m venv .memory_bot
source .memory_bot/bin/activate

pip install -r requirements.txt
```

On Windows, activate the environment with `.memory_bot\Scripts\activate`.

### Start the backend

From the project folder:

```bash
cd Backend
uvicorn app:app --reload --port 8000
```

### Start the frontend

Open another terminal in the project folder:

```bash
source .memory_bot/bin/activate
python Frontend/serve.py
```

The browser opens automatically. The default address is `http://127.0.0.1:5500`; if that port is busy, the launcher selects another port.

Sign in with your username, Groq API key, and Hugging Face token. Use the same username to access your saved memories. Initial model downloads require internet access and may take time.

## How memory works

1. The chatbot uses recent exchanges and relevant saved memories to answer questions.
2. After **five user–assistant exchanges**, it waits 30 seconds and generates a title and detailed summary.
3. Long conversations are summarized in smaller sections to handle API limits.
4. The summary and full original conversation are saved in ChromaDB.
5. Click a memory title to view its summary and conversation. The sidebar refreshes automatically.

Failed saves keep the conversation pending for retry. Signing out also saves fewer than five remaining exchanges. **Unsaved exchanges are lost if the backend restarts**, so sign out before stopping it.

Older memories may contain only a summary if their original conversations were not stored.

## Models

| Purpose | Model |
|---|---|
| Chat and summaries | `openai/gpt-oss-120b` via Groq |
| Evaluation | `openai/gpt-oss-20b` via Groq |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |

Embeddings and reranking run locally on CPU. Saved memories are stored in `Database/chroma_db`.

## Evaluation

Answers are evaluated for correctness, groundedness, relevance, completeness, and instruction compliance. The evaluator uses previous conversations and available evidence; it can report **cannot verify** when evidence is insufficient.

Results are logged in `Evaluation/eval_log.json`. Evaluation failures do not discard the chat answer.

## Tests

Run from the project folder:

```bash
python -m unittest discover -s tests -v
node tests/test_memory_details_ui.cjs
```

Node.js is needed only for the UI checks. Tests use mocked services and do not verify live model quality.

## Notes

- The separate development frontend expects the backend at `http://127.0.0.1:8000`; the container uses the same origin for both.
- Summarization, rate-limit retries, and evaluation can make replies take longer.
- Web-search dates default to `Asia/Singapore`; change them with `MEMORYBOT_TIMEZONE`.
- This is a local prototype. Usernames select memory collections but are not authenticated accounts.

## Docker

Build and start the complete frontend + backend from the project folder:

```bash
docker compose up --build -d
```

Open `http://localhost:8000` and sign in as usual. API keys are entered at runtime;
no keys, local databases, or conversation logs are included in the image. The frontend
uses the API on the same host and port, so a different host port works too:

```bash
MEMORYBOT_PORT=8080 docker compose up -d
```

Named volumes keep saved memories/evaluation logs and downloaded models across container
replacements. Public Hugging Face models download on first database/reranker use; runtime
internet access is still needed for initial downloads, Groq, and web search. The health
check verifies HTTP readiness, not API-key validity or model availability.

Sign out to save pending conversations before stopping the container. Use
`docker compose down` to stop it; adding `-v` also deletes the saved-data volumes.
The app runs as a non-root user with one worker because sessions are held in memory.
The Compose port binds to localhost by default; the application is still a prototype
without account authentication.

After the pipeline publishes an image, run it on an x86-64 or ARM64 machine with Docker
(Linux containers; Docker Desktop on macOS/Windows):

```bash
docker pull ghcr.io/rakshithasrinivas37/ai_memorybot:latest
docker run -d --name memorybot --init \
  -p 127.0.0.1:8000:8000 \
  -v memorybot-data:/data \
  -v memorybot-models:/home/app/.cache/huggingface \
  ghcr.io/rakshithasrinivas37/ai_memorybot:latest
```

Use `:sha-<full-commit-sha>` instead of `:latest` to run a specific published revision.
If the GHCR package is private, authenticate with `docker login ghcr.io` using a token
with `read:packages`, or make the package public in GitHub's package settings.

## CI/CD on GitHub

The pipelines are separate: `.github/workflows/ci.yml` and `.github/workflows/cd.yml`.

1. **Push to a `feature/**` branch:** run Python and frontend tests, then build and smoke-test the Docker image.
2. **Pull request to `main`:** run the same checks on the proposed merge. `CI gate` succeeds only when tests and the container check pass.
3. **Review and merge:** review the pull request and manually merge it into `main` after CI passes. Nothing auto-merges.
4. **Merge to main:** the resulting push triggers CD to build and publish `linux/amd64` and `linux/arm64` images to GitHub Container Registry, tagged `latest` and `sha-<commit>`.

The publish job uses GitHub's built-in `GITHUB_TOKEN` with `packages: write` permission.
Branch and pull-request jobs have read-only repository access and do not publish images.
This is image delivery only; it does not deploy to a server. A feature branch needs `ci.yml` for its push to trigger CI. CD runs only on pushes to
`main`; branch protection requires CI before the manual PR merge. Open a PR manually; pushes do not create one.

### One-time repository setup

- Enable GitHub Actions and allow the actions used in the workflow.
- Add a branch protection rule for `main`: require pull requests, **`CI gate`** status check, an up-to-date branch, and resolved conversations. Enforce the rule for administrators too.
- A separate approving review is not required because this repository has one collaborator.
  Merge remains manual. If you add collaborators, you can require an approving review later.
- After the first publication, choose whether the GHCR package should be public or private.

A ready-to-apply protection configuration is included. With GitHub CLI authenticated
as a repository administrator, run from the project root:

```bash
gh api --method PUT repos/Rakshithasrinivas37/AI_MemoryBot/branches/main/protection \
  --input .github/main-protection.json
```

This command replaces the existing protection configuration; inspect any existing rules
before applying it. Workflow files alone do not require pull requests or block direct pushes.
Push this change on a feature branch and open a PR to introduce the pipeline. The required
check will run when the PR contains the workflow.

For the API routing tests, install `httpx==0.28.1` alongside the project dependencies.
You can reproduce the pipeline's frontend and container checks locally:

```bash
node tests/test_api_config.cjs
docker build -t memorybot:ci .
bash scripts/smoke_container.sh memorybot:ci
```
