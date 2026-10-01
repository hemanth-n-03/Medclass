"""
Precompute Dense Semantic Embeddings for MedQuAD Knowledge Base
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INDEX_DIR = os.path.join(BASE_DIR, "ml", "models", "similarity")
MEDQUAD_PATH = os.path.join(BASE_DIR, "datasets", "medquad.csv")
CORPUS_PATH = os.path.join(INDEX_DIR, "sim_corpus.json")
DENSE_EMBED_PATH = os.path.join(INDEX_DIR, "sim_dense_embeddings.npy")

def main():
    print("Loading MedQuAD corpus...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    questions = [item["question"] for item in corpus]
    print(f"Total questions in corpus: {len(questions)}")

    print("Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Encoding {len(questions)} questions in batches...")
    start_t = time.time()
    embeddings = model.encode(
        questions,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    duration = time.time() - start_t
    print(f"Encoding completed in {duration:.2f}s. Shape: {embeddings.shape}")

    np.save(DENSE_EMBED_PATH, embeddings)
    print(f"Saved dense embeddings to {DENSE_EMBED_PATH}")

if __name__ == "__main__":
    main()
