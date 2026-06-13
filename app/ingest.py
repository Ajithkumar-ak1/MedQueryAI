import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
import time

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-rag")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")



def load_pdf_file(data_path):

    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents



def text_splitter(documents,chunk_size=500,chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
dimension = len(embeddings.embed_query("test"))
print(f"Embedding Dimension: {dimension}")


pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [
    index["name"]
    for index in pc.list_indexes()
]

if INDEX_NAME not in existing_indexes:
    print(f"Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("Pinecone index ready")


documents = load_pdf_file("../Data")
print(f"Loaded {len(documents)} pages")
chunks = text_splitter(documents)
print(f"Created {len(chunks)} chunks")



pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

vectors = []

print("Creating embeddings...")

for i, chunk in enumerate(chunks):

    embedding = embeddings.embed_query(chunk.page_content)

    vectors.append({
        "id": str(uuid.uuid4()),
        "values": embedding,
        "metadata": {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page", -1)
        }
    })

    if i % 500 == 0:
        print(f"Processed {i}/{len(chunks)} chunks")

print(f"Total vectors ready: {len(vectors)}")


BATCH_SIZE = 100

print("Uploading to Pinecone...")

for i in range(0, len(vectors), BATCH_SIZE):

    batch = vectors[i:i + BATCH_SIZE]

    index.upsert(vectors=batch)

    print(f"Uploaded {min(i+BATCH_SIZE, len(vectors))}/{len(vectors)}")

    time.sleep(0.5)

print("Upload complete!")

print(index.describe_index_stats())