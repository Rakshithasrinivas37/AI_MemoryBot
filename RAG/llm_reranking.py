"""Local cross-encoder reranking of retrieved memory candidates."""
from functools import lru_cache
import logging
import math

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker():
    """Load lazily and reuse the model across queries."""
    return CrossEncoder(RERANKER_MODEL, device="cpu", max_length=512)


def llm_rerank(query, memories, client=None):
    """Rank by relevance score; client is retained for caller compatibility.

    If local inference fails, preserve the incoming hybrid-retrieval ranking.
    """
    if len(memories) <= 1:
        return memories

    try:
        scores = get_reranker().predict(
            [(query, memory) for memory in memories],
            show_progress_bar=False,
        )
        if len(scores) != len(memories) or not all(math.isfinite(float(s)) for s in scores):
            raise ValueError("Reranker returned invalid relevance scores")
        # Stable sorting preserves the hybrid order when scores are tied.
        indices = sorted(range(len(memories)), key=lambda i: float(scores[i]), reverse=True)
        return [memories[i] for i in indices]
    except Exception as error:
        logger.warning("Cross-encoder reranking failed; keeping hybrid order: %s", error)
        return memories
