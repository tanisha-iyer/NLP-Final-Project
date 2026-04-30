"""
RAG engine: retrieval from FAISS + generation via Groq and Together AI.
"""

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

#load index 
INDEX_PATH = "faiss_index.bin"
META_PATH  = "faiss_meta.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_index  = None
_chunks = None
_model  = None

def _load():
    global _index, _chunks, _model
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    if _chunks is None:
        with open(META_PATH, "rb") as f:
            _chunks = pickle.load(f)
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)


# retrieval 

def retrieve(query: str, top_k: int = 8) -> list[dict]:
    """Embed query and return top_k most similar chunks."""
    _load()
    q_emb = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    distances, indices = _index.search(q_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = _chunks[idx].copy()
        chunk["score_sim"] = round(float(dist), 4)
        results.append(chunk)
    return results


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    lines = []
    for i, c in enumerate(chunks, 1):
        source = f"[{c['type'].upper()} | r/{c.get('subreddit','?')} | score: {c.get('score',0):.0f}]"
        lines.append(f"{i}. {source}\n   {c['text']}")
    return "\n\n".join(lines)


def build_prompt(query: str, context: str) -> str:
    return f"""You are an expert analyst of Reddit climate change discussions.
Use ONLY the Reddit posts and comments provided below to answer the question.
If the answer cannot be found in the context, say "This information is not present in the retrieved Reddit content."

--- RETRIEVED REDDIT CONTENT ---
{context}
--- END OF CONTEXT ---

Question: {query}

Answer:"""


# LLM 

def call_groq(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant") -> str:
    from groq import Groq
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def call_together(prompt: str, api_key: str,
                  model: str = "meta-llama/Llama-3-8b-chat-hf") -> str:
    """Call Together AI API (free tier).
    Free models include:
      - meta-llama/Llama-3-8b-chat-hf
      - mistralai/Mistral-7B-Instruct-v0.1
      - google/gemma-2b-it
    """
    from together import Together
    client   = Together(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# Main RAG Function 

def rag_answer(query: str, llm: str, api_keys: dict, top_k: int = 8) -> dict:
    chunks  = retrieve(query, top_k=top_k)
    context = format_context(chunks)
    prompt  = build_prompt(query, context)

    if llm == "Groq (LLaMA3)":
        answer = call_groq(prompt, api_keys["groq"])
    elif llm == "Together AI (LLaMA3)":
        answer = call_together(prompt, api_keys["together"])
    else:
        answer = "No valid LLM selected."

    return {"answer": answer, "chunks": chunks, "context": context, "prompt": prompt}