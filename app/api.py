from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from generator import ask_medivault

app = FastAPI(
    title="Medivault AI",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():

    return {
        "message": "Medivault AI API Running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask(request: QueryRequest):

    result = ask_medivault(
        request.query
    )

    return {
        "query": request.query,
        "rewritten_query":
            result.get(
                "rewritten_query",
                request.query
            ),
        "answer":
            result["answer"],
        "sources":
            result["sources"],
        "metrics":
            result.get(
                "metrics",
                {}
            )
    }