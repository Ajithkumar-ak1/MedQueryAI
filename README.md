# 🏥 MedQueryAI: Hybrid Medical RAG System

A production-oriented Retrieval-Augmented Generation (RAG) system for medical question answering that combines semantic search, BM25 lexical retrieval, cross-encoder reranking, and Groq-powered LLM inference to deliver accurate, context-grounded medical responses.

---

## 🚀 Overview

MedQueryAI is designed to answer medical questions using trusted medical documents rather than relying solely on an LLM's internal knowledge.

The system retrieves relevant information from a medical knowledge base, reranks the results for precision, and generates evidence-grounded answers.

### Core Objectives

- Improve factual accuracy
- Reduce hallucinations
- Retrieve medically relevant information
- Provide source-grounded responses
- Deliver fast inference using Groq

---

## ✨ Features

### Advanced Retrieval Pipeline

- Dense Semantic Search using Pinecone
- BM25 Lexical Retrieval
- Hybrid Retrieval Fusion
- Cross-Encoder Reranking
- Context-Aware Retrieval
- Source Tracking

### Medical Question Answering

- Medical Knowledge Retrieval
- Context-Grounded Responses
- Reduced Hallucinations
- Multi-Document Retrieval
- Evidence-Based Generation

### Engineering Features

- FastAPI Backend
- Pinecone Vector Database
- Groq LLM Integration
- Modular Architecture
- Performance Monitoring
- Environment-Based Configuration

---

# 🏗 System Architecture

```text
                    ┌─────────────────────┐
                    │      User Query     │
                    └──────────┬──────────┘
                               │
                               ▼
                 HTML / CSS / JavaScript Frontend
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Query Embedding     │
                    │ all-MiniLM-L6-v2    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Pinecone Retrieval  │
                    │ Top-K Chunks        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ BM25 Re-Scoring     │
                    │ Lexical Matching    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Hybrid Fusion       │
                    │ Dense + BM25        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Cross Encoder       │
                    │ Re-Ranking          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Context Builder     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Groq LLM            │
                    │ Response Generation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Final Answer        │
                    └─────────────────────┘
```

---

# 🔍 Retrieval Pipeline

## 1. Dense Semantic Retrieval

The user query is converted into a vector embedding using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding is searched against Pinecone to retrieve the most semantically relevant chunks.

### Benefits

- Captures semantic meaning
- Handles synonyms
- Understands medical context

---

## 2. BM25 Lexical Retrieval

Retrieved chunks are scored using BM25 to strengthen exact keyword matching.

### Benefits

- Matches exact medical terminology
- Improves acronym retrieval
- Better drug-name matching
- Improves precision for rare terms

---

## 3. Hybrid Retrieval Fusion

The final retrieval score combines semantic similarity and lexical relevance.

```text
Hybrid Score =
0.75 × Dense Score
+
0.25 × BM25 Score
```

### Why Hybrid Search?

Dense retrieval excels at understanding meaning.

BM25 excels at exact keyword matching.

Combining both improves retrieval quality.

---

## 4. Cross-Encoder Re-Ranking

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Each query-document pair is independently evaluated.

### Benefits

- Better relevance estimation
- Improved ranking quality
- Higher answer accuracy

---

## 5. Context Construction

The highest-ranked chunks are merged into a single context block:

```text
Chunk 1

Chunk 2

Chunk 3

Chunk 4

Chunk 5
```

This context is supplied to the LLM.

---

## 6. Response Generation

The LLM receives:

```text
System Prompt
+
Retrieved Context
+
User Query
```

and generates a grounded response based on retrieved evidence.

---

# ⚙ Technology Stack

## AI & Machine Learning

| Component | Technology |
|------------|------------|
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Reranker | Cross Encoder |
| Reranker Model | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM Provider | Groq |
| LLM Models | Llama 3, Gemma, DeepSeek |

---

## Retrieval Layer

| Component | Technology |
|------------|------------|
| Vector Database | Pinecone |
| Dense Retrieval | Cosine Similarity |
| Lexical Retrieval | BM25 |
| Hybrid Retrieval | Dense + BM25 Fusion |
| Re-Ranking | Cross Encoder |

---

## Backend

| Component | Technology |
|------------|------------|
| Framework | FastAPI |
| Server | Uvicorn |
| Environment Management | Python Dotenv |

---

## Frontend

| Component | Technology |
|------------|------------|
| Structure | HTML5 |
| Styling | CSS3 |
| Interactivity | JavaScript |
| API Communication | Fetch API |

---

## Data Processing

| Component | Technology |
|------------|------------|
| PDF Parsing | PyPDF |
| Chunking | Recursive Character Text Splitter |
| Metadata Tracking | Source + Page Number |

---
# 📂 Project Structure

```text
MedQueryAI/
│
├── Data/
│   └── pdf/
│       ├── medical_document_1.pdf
│       ├── medical_document_2.pdf
│       └── ...
│
├── app/
│   ├── api.py                # FastAPI endpoints
│   ├── generator.py          # Groq LLM response generation
│   ├── retriever.py          # Hybrid retrieval pipeline
│   ├── ingest.py             # PDF ingestion & vectorization
│   ├── main.py               # Application entry point
│   ├── index.html            # Frontend UI
│   └── bm25_corpus.pkl       # Precomputed BM25 corpus
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🔄 Application Workflow

```text
PDF Documents
      │
      ▼
Ingestion Pipeline
      │
      ▼
Text Chunking
      │
      ▼
MiniLM Embeddings
      │
      ▼
Pinecone Vector Database
      │
      ▼
───────────────────────────────
           User Query
───────────────────────────────
      │
      ▼
Query Embedding
      │
      ▼
Pinecone Dense Retrieval
      │
      ▼
BM25 Lexical Scoring
      │
      ▼
Hybrid Score Fusion
      │
      ▼
Cross Encoder Reranking
      │
      ▼
Top Relevant Chunks
      │
      ▼
Groq LLM
      │
      ▼
Medical Response
```

---

# 🛠 Technologies Used

## Retrieval & Search

- Pinecone Vector Database
- BM25 Lexical Retrieval
- Hybrid Search Fusion
- Cross-Encoder Reranking

## Machine Learning

- Sentence Transformers
- all-MiniLM-L6-v2
- cross-encoder/ms-marco-MiniLM-L-6-v2

## LLM

- Groq API
- Llama 3 Models

## Backend

- FastAPI
- Uvicorn
- Python

## Frontend

- HTML
- CSS
- JavaScript

## Data Processing

- PyPDF
- Recursive Text Chunking
- Metadata Tracking

---

# 🎯 Key Engineering Highlights

### Hybrid Retrieval Pipeline

Instead of relying only on vector search, MedQueryAI combines:

```text
Dense Retrieval (Semantic Similarity)
+
BM25 Retrieval (Keyword Matching)
+
Cross Encoder Reranking
```

This improves retrieval precision, especially for:

- Medical terminology
- Drug names
- Disease names
- Clinical abbreviations

---

### Performance Optimizations

- Precomputed BM25 Corpus (`bm25_corpus.pkl`)
- Pinecone Vector Search
- Lightweight MiniLM Embeddings
- Groq Ultra-Fast Inference
- Top-K Candidate Reranking

---

### Metrics Tracking

The system records:

```text
embedding_time
retrieval_time
rerank_time
generation_time
total_time
```

Example:

```text
MEDQUERY AI METRICS
=================================
embedding_time : 0.31s
retrieval_time : 2.58s
rerank_time    : 0.49s
generation_time: 0.64s
total_time     : 4.02s
=================================
```
