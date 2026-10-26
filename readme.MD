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

- The frontend expects the backend at `http://127.0.0.1:8000`.
- Summarization, rate-limit retries, and evaluation can make replies take longer.
- Web-search dates default to `Asia/Singapore`; change them with `MEMORYBOT_TIMEZONE`.
- This is a local prototype. Usernames select memory collections but are not authenticated accounts.
