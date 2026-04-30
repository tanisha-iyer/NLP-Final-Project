import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

""" Configuration """

POSTS_PATH    = r"C:\Users\Tanisha Iyer\Downloads\archive (4)\the-reddit-climate-change-dataset-posts.csv"
COMMENTS_PATH = r"C:\Users\Tanisha Iyer\Downloads\archive (4)\the-reddit-climate-change-dataset-comments.csv"
INDEX_PATH    = "faiss_index.bin"
META_PATH     = "faiss_meta.pkl"
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
MAX_POSTS     = 20_000
MAX_COMMENTS  = 30_000
BATCH_SIZE    = 256

def load_and_chunk():
    # Load CSVs then create text chunks with metadata.
    print("Loading...")
    posts = pd.read_csv(POSTS_PATH, usecols=["created_utc", "title", "score", "subreddit.name"])
    posts = posts.dropna(subset=["title"]).sample(
        min(MAX_POSTS, len(posts)), random_state=42
    )

    print("Loading comments...")
    comments = pd.read_csv(COMMENTS_PATH, usecols=["created_utc", "body", "score", "sentiment", "subreddit.name"])
    comments = comments.dropna(subset=["body"])
    comments = comments[comments["body"].str.len() > 50]   # skip very short comments
    comments = comments.sample(min(MAX_COMMENTS, len(comments)), random_state=42)

    chunks = []

    # Post chunks: title is the chunk text
    for _, row in posts.iterrows():
        chunks.append({
            "text":      str(row["title"]),
            "type":      "post",
            "subreddit": str(row.get("subreddit.name", "")),
            "score":     float(row.get("score", 0)),
            "created":   str(row.get("created_utc", "")),
        })

    # Comment chunks: body is the chunk text (truncated to 400 chars)
    for _, row in comments.iterrows():
        chunks.append({
            "text":      str(row["body"])[:400],
            "type":      "comment",
            "subreddit": str(row.get("subreddit.name", "")),
            "score":     float(row.get("score", 0)),
            "sentiment": float(row.get("sentiment", 0)),
            "created":   str(row.get("created_utc", "")),
        })

    print(f"Total chunks: {len(chunks):,}")
    return chunks


def embed_chunks(chunks, model):
    """Embed all chunks in batches, return numpy array."""
    texts = [c["text"] for c in chunks]
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embs  = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs)
        if i % 5000 == 0:
            print(f"  Embedded {i:,} / {len(texts):,}")

    return np.vstack(all_embeddings).astype("float32")


def build_and_save():
    chunks = load_and_chunk()
    model = SentenceTransformer(MODEL_NAME)

    embeddings = embed_chunks(chunks, model)

    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)

    print("Building FAISS index")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalisation
    index.add(embeddings)

    print(f"Saving index ({index.ntotal:,} vectors) to {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)


if __name__ == "__main__":
    build_and_save()

