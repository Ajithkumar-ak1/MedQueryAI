import os
import re
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
load_dotenv()



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


def keyword_score(query, text):
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    return len(query_words.intersection(text_words))


def hybrid_search(query,top_k=10,rerank_top_k=5):
    metrics = {}
    start = time.perf_counter()

    query_embedding = embeddings.embed_query(query)
    metrics["embedding_time"] = (time.perf_counter() - start)

    start = time.perf_counter()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    metrics["retrieval_time"] = (time.perf_counter() - start)
    candidates = []

    for match in results["matches"]:

        text = match["metadata"].get(
            "text",
            ""
        )
        dense_score = float(match["score"])
        sparse_score = keyword_score(query,text)
        hybrid_score = (0.8 * dense_score +0.2 * sparse_score)

        candidates.append({
            "text": text,
            "source": match["metadata"].get(
                "source",
                "unknown"
            ),
            "page": match["metadata"].get(
                "page",
                -1
            ),
            "dense_score": dense_score,
            "keyword_score": sparse_score,
            "hybrid_score": hybrid_score
        })

    candidates.sort(key=lambda x: x["hybrid_score"],reverse=True)
    candidates = candidates[:10]

    start = time.perf_counter()
    pairs = [
        (query, doc["text"])
        for doc in candidates
    ]

    rerank_scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    metrics["rerank_time"] = (
        time.perf_counter() - start
    )

    for doc, score in zip(candidates,rerank_scores):
        doc["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"],reverse=True)
    return candidates[:rerank_top_k],  metrics


def retrieve_context(query,top_k=10,rerank_top_k=5):
    docs, metrics = hybrid_search(
        query=query,
        top_k=top_k,
        rerank_top_k=rerank_top_k
    )

    context = "\n\n".join(
        doc["text"]
        for doc in docs
    )

    return context, docs, metrics   