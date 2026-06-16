import os
import re
import uuid
import time
import hashlib
import pickle
import logging
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "medical-rag")
PDF_PATH         = os.getenv("PDF_PATH", "../Data/document.pdf")   # single file
BM25_CACHE_PATH  = os.getenv("BM25_CACHE_PATH", "bm25_corpus.pkl")

# Larger chunks + more overlap for dense medical prose
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE",    1000))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP",  200))  # 20 % overlap

EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE",  64))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", 100))
UPSERT_SLEEP      = float(os.getenv("UPSERT_SLEEP", 0.3))

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment / .env")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Whitespace+punctuation tokenizer — must match retrieval code exactly."""
    return re.findall(r"\w+", text.lower())


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_pdf(pdf_path: str):
    """
    Load a single PDF with PyPDFLoader (one Document per page).
    Fails loudly if the file is missing — better than a silent empty corpus.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    log.info("Loaded %d pages from '%s'", len(pages), path.name)
    return pages


def extract_section_heading(text: str) -> str:
    """
    Heuristic: treat the first non-empty line as the section heading if it is
    short (≤ 80 chars) and doesn't end with a period (i.e. looks like a title).
    Falls back to empty string — never crashes.
    """
    for line in text.splitlines():
        line = line.strip()
        if line and len(line) <= 80 and not line.endswith("."):
            return line
    return ""


def split_pages(pages, chunk_size: int, chunk_overlap: int):
    """
    Split with paragraph-aware separators.
    Medical PDFs commonly use double newlines between sections — try those first
    so chunks respect section boundaries before falling back to sentences.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    log.info(
        "Split into %d chunks (size=%d, overlap=%d)",
        len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def deduplicate(chunks) -> list:
    """Remove exact-duplicate chunks (catches repeated ToC / header pages)."""
    seen: set[str] = set()
    unique = []
    for chunk in chunks:
        h = content_hash(chunk.page_content)
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    if removed:
        log.info("Removed %d exact-duplicate chunks", removed)
    return unique


def batch_embed(texts: list[str], model, batch_size: int) -> list[list[float]]:
    """
    Use embed_documents (passage mode) in batches.
    Never use embed_query for indexing — it applies query-side prompt prefixes.
    """
    all_vecs: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start: start + batch_size]
        vecs  = model.embed_documents(batch)
        all_vecs.extend(vecs)
        log.info("Embedded %d / %d", min(start + batch_size, total), total)
    return all_vecs


def build_and_save_bm25(chunks, cache_path: str) -> None:
    """
    Build BM25 over the FULL corpus and pickle it.

    With a single 600-page PDF this is ~1 000–2 000 chunks — small enough to
    fit in RAM and give BM25 meaningful IDF scores across the whole document.
    The retrieval code loads this once at startup instead of rebuilding over
    10 Pinecone candidates per query.

    Stored payload:
        bm25    – BM25Okapi object (queryable)
        corpus  – list of raw chunk texts (index-aligned with bm25)
        meta    – list of {source, page, chunk_index} dicts (same alignment)
    """
    corpus = [tokenize(c.page_content) for c in chunks]
    bm25   = BM25Okapi(corpus)
    payload = {
        "bm25":   bm25,
        "corpus": [c.page_content for c in chunks],
        "meta": [
            {
                "source":      c.metadata.get("source", "unknown"),
                "page":        c.metadata.get("page", -1),
                "chunk_index": i,
            }
            for i, c in enumerate(chunks)
        ],
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    log.info("BM25 corpus saved → %s (%d docs)", cache_path, len(corpus))


def build_vectors(chunks, embeddings_list: list) -> list[dict]:
    """
    Pair each chunk with its embedding and build the Pinecone record.

    Extra metadata stored per vector:
        section_heading  – heuristic first-line title (useful for citations)
        chunk_index      – position in the full chunk list (for context stitching)
        prev_page / next_page – neighbouring pages (lets retrieval fetch adjacent
                                chunks when an answer spans a boundary)
        char_count       – lets retrieval down-weight very short fragments
    """
    vectors = []
    total   = len(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
        page    = chunk.metadata.get("page", -1)
        source  = chunk.metadata.get("source", "unknown")
        heading = extract_section_heading(chunk.page_content)

        vectors.append({
            "id":     str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text":            chunk.page_content,
                "source":          source,
                "page":            page,
                "prev_page":       page - 1 if page > 0   else -1,
                "next_page":       page + 1 if page >= 0  else -1,
                "chunk_index":     i,
                "total_chunks":    total,
                "section_heading": heading,
                "char_count":      len(chunk.page_content),
            },
        })
    return vectors


def ensure_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        log.info("Creating index '%s' (dim=%d)", index_name, dimension)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        for _ in range(20):
            if pc.describe_index(index_name)["status"].get("ready"):
                break
            time.sleep(2)
    log.info("Index '%s' is ready", index_name)


def upsert_vectors(index, vectors: list[dict], batch_size: int, sleep: float) -> None:
    total = len(vectors)
    for start in range(0, total, batch_size):
        batch    = vectors[start: start + batch_size]
        index.upsert(vectors=batch)
        uploaded = min(start + batch_size, total)
        log.info("Upserted %d / %d", uploaded, total)
        if uploaded < total:
            time.sleep(sleep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load
    pages  = load_pdf(PDF_PATH)

    # 2. Split + deduplicate
    chunks = split_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = deduplicate(chunks)

    # 3. Persist BM25 over the full corpus — used at query time
    build_and_save_bm25(chunks, BM25_CACHE_PATH)

    # 4. Embed (passage mode, batched)
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    dimension = len(embed_model.embed_query("test"))
    log.info("Embedding dimension: %d", dimension)

    texts          = [c.page_content for c in chunks]
    embeddings_list = batch_embed(texts, embed_model, EMBED_BATCH_SIZE)

    # 5. Build vector records
    vectors = build_vectors(chunks, embeddings_list)
    log.info("Prepared %d vectors", len(vectors))

    # 6. Upsert to Pinecone
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, INDEX_NAME, dimension)
    index = pc.Index(INDEX_NAME)
    upsert_vectors(index, vectors, UPSERT_BATCH_SIZE, UPSERT_SLEEP)

    stats = index.describe_index_stats()
    log.info("Done. Index stats: %s", stats)


if __name__ == "__main__":
    main()