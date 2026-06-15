import os
import re
import time
from dotenv import load_dotenv
from groq import Groq
from retriever import retrieve_context
load_dotenv()


client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"

chat_history = []
MAX_HISTORY = 5

SYSTEM_PROMPT = """
You are Medivault AI, a professional medical assistant.

Rules:
- Answer ONLY from the provided medical context.
- Do not make up information.
- If the answer is not present in the context, say:
  "I could not find sufficient information in the medical knowledge base."
- Keep answers concise, accurate, and easy to understand.
- Use bullet points when appropriate.
- Never reveal internal reasoning.
- Never output <think> tags.
"""


def clean_response(text):

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    return text.strip()

def format_history():

    if not chat_history:
        return "No previous conversation."

    history_text = ""

    for turn in chat_history:

        history_text += f"""
User: {turn['user']}
Assistant: {turn['assistant']}
"""

    return history_text

def rewrite_query(query):

    history = format_history()

    prompt = f"""
You are a query rewriting assistant.

Your task:
Rewrite the user's latest question into a complete standalone question.

Rules:
- Resolve references like:
  it, its, they, them, this, that, these, those.
- Use conversation history.
- Preserve meaning.
- Return ONLY the rewritten question.
- If already standalone, return it unchanged.

Conversation History:
{history}

Latest Question:
{query}

Standalone Question:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    rewritten_query = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    return rewritten_query

def generate_answer(query, context):

    prompt = f"""
Medical Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content
    return clean_response(answer)


def ask_medivault(query):
    total_start = time.perf_counter()
    rewritten_query = rewrite_query(query)

    print(f"\nRewritten Query: "
f"{rewritten_query}"
    )

    context, docs, retrieval_metrics = retrieve_context(
        rewritten_query
    )

    generation_start = time.perf_counter()
    answer = generate_answer(
        rewritten_query,
        context
    )

    generation_time = (time.perf_counter() - generation_start)
    total_time = (time.perf_counter() - total_start)

    chat_history.append({
        "user": query,
        "assistant": answer
    })

    if len(chat_history) > MAX_HISTORY:
        chat_history.pop(0)

    sources = []
    seen = set()

    for doc in docs:
        key = (doc["source"],doc["page"])

        if key not in seen:
            seen.add(key)
            sources.append({
                "source": doc["source"],
                "page": doc["page"]
            })

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources": sources,
        "metrics": {
            **retrieval_metrics,
            "generation_time": generation_time,
            "total_time": total_time
        }
    }