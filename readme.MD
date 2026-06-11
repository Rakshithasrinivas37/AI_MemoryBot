# 🧠 AI MemoryBot

An AI-powered chatbot with persistent, per-user long-term memory. Unlike traditional chatbots that forget everything after a session, AI MemoryBot remembers facts about your past conversations and recalls them intelligently whenever its necessary.

---

## ✨ Features

- **Persistent Memory** — Remembers user details across sessions using a vector database
- **Hybrid Memory Retrieval** — Combines vector similarity, keyword ranking, and LLM reranking for accurate recall
- **Two-Tier Memory Architecture** — Short-term (in-session context) + Long-term (summarized & stored)
- **Agentic Tool Use** — Automatically use web search tool to fetch current information and to fetch current date
- **Memory Management** — View and delete your stored memories via the UI
- **Multi-user Support** — Isolated memory per user with session management

---

## ⚙️ How It Works

### Memory Pipeline

1. **Short-Term Memory** — Keeps the last 5 message pairs at the active context window
2. **Auto-Summarization** — For every 5 messages, a summary is generated using LLaMA 3.3 70B and stored in ChromaDB
3. **Retrieval (4-stage):**
   - Vector similarity search
   - Keyword overlap ranking
   - Weighted combined scoring
   - LLM reranking for final relevance ordering
4. **Prompt Injection** — Top-ranked memories are injected into the system prompt before each response

### Agentic Tools

The bot automatically decides when to use tools:
- `web_search` — triggered for queries about latest/recent/current information
- `get_current_date` — triggered when the user asks about today's date

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com/)
- [HuggingFace Token](https://huggingface.co/settings/tokens)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-memorybot.git
cd ai-memorybot

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
# Start the backend
uvicorn Backend.app:app --reload --port 8000

# Start the frontend (in a separate terminal)
python Frontend/serve.py
```

Then open your browser at `http://localhost:5500` and log in with your username, Groq API key and HF Token.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | LLaMA 3.3 70B / GPT-OSS 120B via Groq API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB | ChromaDB |
| Backend | FastAPI (Python) |
| Frontend | HTML, CSS, JavaScript |
| Search Tool | DuckDuckGo Search (ddgs) |

---