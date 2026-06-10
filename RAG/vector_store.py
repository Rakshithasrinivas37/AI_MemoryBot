from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import Chroma
import uuid
from datetime import datetime
from RAG.llm_reranking import llm_rerank

import os

# ── Get root path dynamically ──────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Stores ChromaDB in AI Chatbot/Database/chroma_db
CHROMA_PATH = os.path.join(ROOT_DIR, "Database", "chroma_db")

# ✅ Create folder if it doesn't exist
os.makedirs(CHROMA_PATH, exist_ok=True)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


EMBEDDINGS = HuggingFaceEmbeddings(
    model_name    = MODEL_ID,
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True}
)

def get_db(username):
    return Chroma(
      collection_name    = f"long_term_memory_{username}",
      embedding_function = EMBEDDINGS,
      persist_directory  = CHROMA_PATH,
      collection_metadata = {"hnsw:space": "cosine"}
  )


def store_in_database(long_term_memory, username):
    """Store summary — only call when memory is ready."""

    # ✅ Guard — don't store empty memory
    if not long_term_memory or long_term_memory.strip() == "":
        print("⚠️ Memory is empty — nothing stored")
        return

    vectorstore = get_db(username)
    vectorstore.add_texts(
        texts     = [long_term_memory],
        metadatas = [{
            "source": "long_term_memory",
            "username": username,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }],
        ids       = [str(uuid.uuid4())]
    )
    print(f"✅ Stored in DB: {long_term_memory[:60]}...")

def retrieve_from_database(query, client, username, top_k=3):
    ## Retrieve from database and rank the retrieved responses
    vectorstore = get_db(username)
    existing    = vectorstore.get()

    if not existing["documents"]:
        return ""

    total = len(existing["documents"])
    print(f"\n📂 Total memories: {total}")

    # ── Step 1: Vector similarity ──────────────────────
    results = vectorstore.similarity_search_with_score(
        query = query,
        k     = total
    )
    results.sort(key=lambda x: x[1])

    print("\n📊 Step 1 — Similarity ranking:")
    for i, (doc, score) in enumerate(results):
        relevance = round((1 - score) * 100, 1)
        print(f"  [{i+1}] {relevance}% | {doc.page_content[:60]}")

    memories = [doc.page_content for doc, score in results]

    # ── Step 2: Keyword ranking ────────────────────────
    stop_words = {"i", "am", "is", "are", "the", "a",
                  "an", "my", "me", "what", "how", "who",
                  "do", "did", "was", "were", "have", "has",
                  "you", "your", "we", "our", "they"}

    # ✅ Strip punctuation before filtering stop words
    query_words = set([
        w.lower().strip("?.,!\"'")
        for w in query.split()
        if w.lower().strip("?.,!\"'") not in stop_words
        and w.lower().strip("?.,!\"'") != ""
    ])

    print(f"\n📊 Step 2 — Keyword ranking:")
    print(f"  Keywords: {query_words}")

    keyword_scores = []
    for memory in memories:
        memory_words = set(memory.lower().split())
        overlap      = query_words & memory_words
        keyword_scores.append((memory, len(overlap), overlap))
        print(f"  Score={len(overlap)} | "
              f"Matched={overlap} | {memory[:50]}")

    # ── Step 3: Weighted combined score ───────────────
    # similarity rank weight = 1
    # keyword score weight   = 2 ✅ keyword more important
    sim_ranks = {m: i for i, m in enumerate(memories)}
    kw_sorted = sorted(
        keyword_scores,
        key=lambda x: x[1],
        reverse=True
    )
    kw_ranks  = {m: i for i, (m, s, o) in enumerate(kw_sorted)}

    print(f"\n📊 Step 3 — Weighted combined ranking:")
    combined_scores = {}
    for memory in memories:
        sim_rank       = sim_ranks.get(memory, 0)
        kw_rank        = kw_ranks.get(memory, 0)
        # ✅ Lower combined score = better rank
        combined_score = (sim_rank * 1) + (kw_rank * 2)
        combined_scores[memory] = combined_score
        print(f"  sim_rank={sim_rank} kw_rank={kw_rank} "
              f"combined={combined_score} | {memory[:50]}")

    sorted_memories = sorted(
        combined_scores.keys(),
        key=lambda m: combined_scores[m]
    )

    print(f"\n  Final order after combining:")
    for i, m in enumerate(sorted_memories):
        print(f"  [{i+1}] {m[:60]}")

    # ── Step 4: LLM reranking ──────────────────────────
    top_candidates = sorted_memories[:min(5, len(sorted_memories))]

    print(f"\n📊 Step 4 — LLM reranking:")
    final_ranked   = llm_rerank(query, top_candidates, client)

    for i, m in enumerate(final_ranked):
        print(f"  [{i+1}] {m[:60]}")

    return "\n".join([
        f"Memory {i+1}: {m}"
        for i, m in enumerate(final_ranked[:top_k])
    ])