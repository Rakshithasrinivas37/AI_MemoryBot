from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import Chroma
import uuid
import json
from datetime import datetime
from RAG.llm_reranking import llm_rerank
from RAG.hybrid_ranking import tokenize
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

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


def store_in_database(long_term_memory, username, raw_user_messages="", *, title="", conversations=None):
    """Store summary (searchable by vector) plus the user's raw lines in metadata
    (used only for BM25 keyword ranking). Only call when memory is ready."""

    # ✅ Guard — don't store empty memory
    if not long_term_memory or long_term_memory.strip() == "":
        print("⚠️ Memory is empty — nothing stored")
        return False

    vectorstore = get_db(username)
    vectorstore.add_texts(
        texts     = [long_term_memory],
        metadatas = [{
            "source": "long_term_memory",
            "username": username,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "raw_user_messages": raw_user_messages or "",
            "title": title,
            "conversations_json": json.dumps(conversations or [], ensure_ascii=False)
        }],
        ids       = [str(uuid.uuid4())]
    )
    print(f"✅ Stored in DB: {long_term_memory[:60]}...")
    return True

def retrieve_from_database(query, client, username, top_k=3, rrf_k=60):
    ## Hybrid retrieval: vector search + BM25 (rank_bm25), merged with RRF
    ## (LangChain EnsembleRetriever), then cross-encoder rerank
    vectorstore = get_db(username)
    existing    = vectorstore.get()

    if not existing["documents"]:
        return ""

    documents = existing["documents"]
    metadatas = existing.get("metadatas") or [None] * len(documents)
    total     = len(documents)
    print(f"\n📂 Total memories: {total}")

    # ── Vector retriever (searches the summaries) ──────
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": total})

    # ── BM25 retriever ─────────────────────────────────
    # Index = summary + the user's raw lines (older memories: summary only),
    # but the returned documents carry the summary, so both retrievers return
    # the same text and the ensemble can merge them.
    keyword_texts = [
        f"{doc} {(meta or {}).get('raw_user_messages', '')}".strip()
        for doc, meta in zip(documents, metadatas)
    ]
    bm25_index  = BM25Okapi([tokenize(t) for t in keyword_texts])
    query_terms = tokenize(query)
    bm25_scores = bm25_index.get_scores(query_terms)

    # BM25Okapi.get_top_n pads results with zero-score documents, which would
    # add noise to the fusion. Keep only documents that actually matched.
    n_matches = sum(1 for s in bm25_scores if s > 0)
    print(f"\n📊 BM25 (keywords: {query_terms}) — {n_matches} matching memories")

    if n_matches == 0:
        print("  (no keyword matches — using vector ranking only)")
        fused_docs = vector_retriever.invoke(query)
    else:
        bm25_retriever = BM25Retriever(
            vectorizer      = bm25_index,
            docs            = [Document(page_content=d) for d in documents],
            k               = n_matches,
            preprocess_func = tokenize,
        )
        # ── Reciprocal Rank Fusion (weighted RRF, c = rrf_k) ──
        ensemble = EnsembleRetriever(
            retrievers = [vector_retriever, bm25_retriever],
            weights    = [1.0, 1.0],
            c          = rrf_k,
        )
        fused_docs = ensemble.invoke(query)

    print(f"\n📊 RRF fused ranking (c={rrf_k}):")
    for rank, doc in enumerate(fused_docs, start=1):
        print(f"  [{rank}] {doc.page_content[:60]}")

    sorted_memories = [doc.page_content for doc in fused_docs]

    # ── Step 4: Cross-encoder reranking ──────────────────────────
    top_candidates = sorted_memories[:min(5, len(sorted_memories))]

    print(f"\n📊 Step 4 — Cross-encoder reranking:")
    final_ranked   = llm_rerank(query, top_candidates)

    for i, m in enumerate(final_ranked):
        print(f"  [{i+1}] {m[:60]}")

    return "\n".join([
        f"Memory {i+1}: {m}"
        for i, m in enumerate(final_ranked[:top_k])
    ])