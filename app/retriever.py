import os
import re
import time
import pickle
import logging
import threading
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-safe model initialisation (loaded once, reused across all calls)
# ---------------------------------------------------------------------------

_model_lock     = threading.Lock()
_embed_model    = None
_reranker_model = None

def _get_models():
    global _embed_model, _reranker_model
    if _embed_model is None:
        with _model_lock:
            if _embed_model is None:
                log.info("Loading embedding model…")
                _embed_model = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                log.info("Loading reranker model…")
                _reranker_model = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
    return _embed_model, _reranker_model


# ---------------------------------------------------------------------------
# Pinecone client (single instance)
# ---------------------------------------------------------------------------

_pinecone_lock = threading.Lock()
_pinecone_index = None

def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        with _pinecone_lock:
            if _pinecone_index is None:
                api_key    = os.getenv("PINECONE_API_KEY")
                index_name = os.getenv("PINECONE_INDEX_NAME")
                if not api_key:
                    raise ValueError("PINECONE_API_KEY not set")
                if not index_name:
                    raise ValueError("PINECONE_INDEX_NAME not set")
                _pinecone_index = Pinecone(api_key=api_key).Index(index_name)
    return _pinecone_index


# ---------------------------------------------------------------------------
# Full-corpus BM25 (built by ingest.py, loaded once here)
# ---------------------------------------------------------------------------

_bm25_lock    = threading.Lock()
_bm25_payload = None   # {"bm25": BM25Okapi, "corpus": [...], "meta": [...]}

def _get_bm25():
    global _bm25_payload
    if _bm25_payload is None:
        with _bm25_lock:
            if _bm25_payload is None:
                cache_path = os.getenv("BM25_CACHE_PATH", "bm25_corpus.pkl")
                if not os.path.exists(cache_path):
                    log.warning(
                        "BM25 cache not found at '%s'. "
                        "Run ingest.py first. BM25 scoring will be skipped.",
                        cache_path,
                    )
                else:
                    with open(cache_path, "rb") as f:
                        _bm25_payload = pickle.load(f)
                    log.info(
                        "BM25 corpus loaded from '%s' (%d docs)",
                        cache_path,
                        len(_bm25_payload["corpus"]),
                    )
    return _bm25_payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Must match ingest.py tokenizer exactly."""
    return re.findall(r"\w+", text.lower())


def _minmax(scores: list[float]) -> list[float]:
    """
    Min-max normalise a list of scores to [0, 1].
    Returns all-zeros if the range is zero (avoids division by zero).
    """
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _bm25_scores_for_candidates(
    query_tokens: list[str],
    candidates: list[dict],
    bm25_payload: dict,
) -> list[float]:
    """
    Look up each candidate in the full-corpus BM25 by chunk_index metadata.
    Falls back to querying a mini BM25 built from candidates only when the
    corpus pickle is unavailable or a chunk_index is missing.

    Full-corpus path: IDF is computed over all ~1000-2000 chunks → meaningful.
    Fallback path:    IDF computed over 10 candidates → nearly flat, but still
                      better than no keyword signal at all.
    """
    if bm25_payload is not None:
        bm25   = bm25_payload["bm25"]
        corpus = bm25_payload["corpus"]

        # Map candidate chunk_index → position in BM25 corpus
        all_scores    = bm25.get_scores(query_tokens)   # scores for every doc
        result_scores = []
        for doc in candidates:
            idx = doc.get("chunk_index", -1)
            if 0 <= idx < len(all_scores):
                result_scores.append(float(all_scores[idx]))
            else:
                # chunk_index missing from metadata — score 0
                result_scores.append(0.0)
        return result_scores

    # Fallback: mini BM25 on candidates only
    log.debug("BM25 fallback: scoring over %d candidates only", len(candidates))
    from rank_bm25 import BM25Okapi
    mini_corpus  = [tokenize(doc["text"]) for doc in candidates]
    mini_bm25    = BM25Okapi(mini_corpus)
    return [float(s) for s in mini_bm25.get_scores(query_tokens)]


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = 10,
    rerank_top_k: int = 5,
    dense_weight: float = 0.75,
    bm25_weight: float  = 0.25,
) -> tuple[list[dict], dict]:
    """
    1. Dense retrieval via Pinecone (cosine similarity).
    2. BM25 scoring from the full-corpus pickle built at ingest time.
    3. Hybrid score = dense_weight × norm(dense) + bm25_weight × norm(bm25).
       Both scores are min-max normalised to [0,1] before blending so the
       weights mean what they say regardless of raw score ranges.
    4. Cross-encoder reranking on the top candidates.

    Returns (ranked_docs, metrics).
    """
    metrics: dict = {}
    embed_model, reranker = _get_models()
    index                 = _get_index()
    bm25_payload          = _get_bm25()

    # 1. Embed query
    try:
        t0 = time.perf_counter()
        query_embedding = embed_model.embed_query(query)
        metrics["embedding_time"] = time.perf_counter() - t0
    except Exception as exc:
        log.error("Embedding failed: %s", exc)
        raise

    # 2. Dense retrieval
    try:
        t0 = time.perf_counter()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )
        metrics["retrieval_time"] = time.perf_counter() - t0
    except Exception as exc:
        log.error("Pinecone query failed: %s", exc)
        raise

    candidates = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        candidates.append({
            "text":            meta.get("text", ""),
            "source":          meta.get("source", "unknown"),
            "page":            meta.get("page", -1),
            "section_heading": meta.get("section_heading", ""),
            "chunk_index":     int(meta.get("chunk_index", -1)),
            "dense_score":     float(match["score"]),
        })

    if not candidates:
        log.warning("Pinecone returned no matches for query: %r", query[:80])
        return [], metrics

    # 3. BM25 scoring (full corpus) + hybrid blend
    query_tokens = tokenize(query)
    raw_bm25     = _bm25_scores_for_candidates(query_tokens, candidates, bm25_payload)

    # Normalise both score lists to [0,1] before blending
    dense_scores = _minmax([d["dense_score"] for d in candidates])
    bm25_scores  = _minmax(raw_bm25)

    for i, doc in enumerate(candidates):
        doc["bm25_score"]   = bm25_scores[i]
        doc["hybrid_score"] = (
            dense_weight * dense_scores[i] + bm25_weight * bm25_scores[i]
        )

    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # 4. Rerank
    try:
        t0     = time.perf_counter()
        pairs  = [(query, doc["text"]) for doc in candidates]
        scores = reranker.predict(pairs, show_progress_bar=False)
        metrics["rerank_time"] = time.perf_counter() - t0
    except Exception as exc:
        log.error("Reranker failed: %s", exc)
        raise

    for doc, score in zip(candidates, scores):
        doc["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    slow_threshold_ms = 500
    for key in ("embedding_time", "retrieval_time", "rerank_time"):
        ms = metrics.get(key, 0) * 1000
        if ms > slow_threshold_ms:
            log.warning("%s took %.0f ms (threshold %d ms)", key, ms, slow_threshold_ms)

    return candidates[:rerank_top_k], metrics


# ---------------------------------------------------------------------------
# Public retrieval interface
# ---------------------------------------------------------------------------

def retrieve_context(
    query: str,
    top_k: int = 10,
    rerank_top_k: int = 5,
    dense_weight: float = 0.75,
    bm25_weight: float  = 0.25,
) -> tuple[str, list[dict], dict]:
    """
    Returns (context_string, docs, metrics).

    context_string has source citations inline so the LLM can reference them:

        [Page 42 — Dosage Guidelines]
        <chunk text>

        [Page 43 — Dosage Guidelines]
        <chunk text>
    """
    docs, metrics = hybrid_search(
        query=query,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )

    parts = []
    for doc in docs:
        heading = doc.get("section_heading", "").strip()
        page    = doc.get("page", -1)

        if heading and page >= 0:
            label = f"[Page {page} — {heading}]"
        elif page >= 0:
            label = f"[Page {page}]"
        else:
            label = "[Source unknown]"

        parts.append(f"{label}\n{doc['text']}")

    context = "\n\n".join(parts)
    return context, docs, metrics