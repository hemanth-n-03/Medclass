"""
Semantic Similarity & Clinical Knowledge Retrieval Engine
Combines dense semantic embeddings (Sentence-Transformers all-MiniLM-L6-v2)
with sparse TF-IDF and deterministic intent/entity reranking over verified NIH MedQuAD questions.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MEDQUAD_PATH = os.path.join(DATA_DIR, "medquad.csv")
INDEX_DIR = os.path.join(BASE_DIR, "ml", "models", "similarity")
DENSE_EMBED_PATH = os.path.join(INDEX_DIR, "sim_dense_embeddings.npy")
os.makedirs(INDEX_DIR, exist_ok=True)


class SimilarityEngine:
    def __init__(self):
        self.vectorizer = None
        self.nn_model = None
        self.corpus = []
        self.vectors = None
        self.dense_embeddings: np.ndarray = None
        self.dense_encoder: SentenceTransformer = None
        self.is_loaded = False

    def build_and_save(self, sample_size=8000):
        print(f"Loading MedQuAD for similarity indexing from {MEDQUAD_PATH}...")
        df = pd.read_csv(MEDQUAD_PATH)
        df = df.dropna(subset=["question", "answer"]).drop_duplicates(subset=["question"]).copy()
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        print(f"Indexing {len(df)} curated medical questions...")
        self.corpus = []
        texts = []
        for _, row in df.iterrows():
            q = str(row["question"]).strip()
            ans = str(row["answer"]).strip()
            snippet = ans[:220] + "..." if len(ans) > 220 else ans

            src_raw = str(row.get("source", "NIH"))
            src_map = {
                "NINDS": "NIH / NINDS (Neurological Disorders)",
                "NHLBI": "NIH / NHLBI (Heart, Lung, Blood)",
                "NIDDK": "NIH / NIDDK (Diabetes & Digestive)",
                "CancerGov": "NIH / NCI (National Cancer Institute)",
                "CDC": "CDC (Centers for Disease Control)",
                "GHR": "NIH / GHR (Genetics Reference)",
                "GARD": "NIH / GARD (Rare Diseases)",
                "MPlusHealthTopics": "MedlinePlus (National Library of Medicine)",
                "NIHSeniorHealth": "NIH Senior Health"
            }
            source_org = src_map.get(src_raw, f"NIH / {src_raw}")

            entry = {
                "question": q,
                "answer_snippet": snippet,
                "source": src_raw,
                "source_organization": source_org,
                "focus_area": str(row.get("focus_area", "General Medicine"))
            }
            self.corpus.append(entry)
            texts.append(q)

        # 1. Sparse TF-IDF Index
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=15000,
            sublinear_tf=True
        )
        self.vectors = self.vectorizer.fit_transform(texts)
        self.nn_model = NearestNeighbors(n_neighbors=25, metric="cosine")
        self.nn_model.fit(self.vectors)

        # 2. Dense Semantic Index
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dense_embeddings = encoder.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Save artifacts
        joblib.dump(self.vectorizer, os.path.join(INDEX_DIR, "sim_vectorizer.joblib"))
        joblib.dump(self.nn_model, os.path.join(INDEX_DIR, "sim_nn.joblib"))
        with open(os.path.join(INDEX_DIR, "sim_corpus.json"), "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2)
        np.save(DENSE_EMBED_PATH, self.dense_embeddings)

        print(f"Similarity index successfully saved to {INDEX_DIR}")

    def load(self):
        if self.is_loaded:
            return

        vec_path = os.path.join(INDEX_DIR, "sim_vectorizer.joblib")
        nn_path = os.path.join(INDEX_DIR, "sim_nn.joblib")
        corpus_path = os.path.join(INDEX_DIR, "sim_corpus.json")

        if not (os.path.exists(vec_path) and os.path.exists(nn_path) and os.path.exists(corpus_path)):
            self.build_and_save()

        self.vectorizer = joblib.load(vec_path)
        self.nn_model = joblib.load(nn_path)
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        # Load dense embeddings
        if os.path.exists(DENSE_EMBED_PATH):
            self.dense_embeddings = np.load(DENSE_EMBED_PATH)
        else:
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [item["question"] for item in self.corpus]
            self.dense_embeddings = encoder.encode(
                texts,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            np.save(DENSE_EMBED_PATH, self.dense_embeddings)

        self.dense_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.is_loaded = True

    def search(
        self,
        query: str,
        top_k: int = 4,
        target_intent: str = None,
        target_entities: list = None
    ) -> list:
        if not self.is_loaded:
            self.load()

        if not query or len(query.strip()) < 2:
            return []

        clean_query = query.strip()
        search_query = clean_query
        if target_entities:
            search_query = f"{clean_query} {' '.join(target_entities)}"

        candidate_indices = set()
        dense_scores_dict = {}

        # 1. Dense Semantic Scoring
        if self.dense_embeddings is not None and self.dense_encoder is not None:
            q_emb = self.dense_encoder.encode([search_query], normalize_embeddings=True, show_progress_bar=False)[0]
            sims = np.dot(self.dense_embeddings, q_emb)
            # Pick top 35 dense matches
            top_dense_indices = np.argsort(sims)[::-1][:35]
            for idx in top_dense_indices:
                candidate_indices.add(int(idx))
                dense_scores_dict[int(idx)] = float(sims[idx])

        # 2. Sparse TF-IDF Scoring
        tfidf_scores_dict = {}
        q_vec = self.vectorizer.transform([clean_query])
        if q_vec.nnz > 0:
            distances, indices = self.nn_model.kneighbors(q_vec, n_neighbors=min(25, len(self.corpus)))
            for dist, idx in zip(distances[0], indices[0]):
                candidate_indices.add(int(idx))
                tfidf_scores_dict[int(idx)] = max(0.0, 1.0 - float(dist))

        query_tokens = set(re.findall(r"\w+", clean_query.lower()))
        entity_tokens = set()
        if target_entities:
            for ent in target_entities:
                entity_tokens.update(re.findall(r"\w+", ent.lower()))

        candidates = []
        for idx in candidate_indices:
            item = self.corpus[idx]
            cand_q = item["question"].lower()
            cand_tokens = set(re.findall(r"\w+", cand_q))

            d_score = dense_scores_dict.get(idx, 0.0)
            t_score = tfidf_scores_dict.get(idx, 0.0)

            # Blended base similarity (dense 70%, sparse 30%)
            if d_score > 0 and t_score > 0:
                base_sim = 0.70 * d_score + 0.30 * t_score
            elif d_score > 0:
                base_sim = d_score
            else:
                base_sim = t_score

            # Intent alignment boost
            intent_boost = 0.0
            if target_intent:
                intent_lower = target_intent.lower()
                if intent_lower in cand_q:
                    intent_boost += 0.08
                elif intent_lower == "causes" and any(k in cand_q for k in ["cause", "why", "origin", "risk"]):
                    intent_boost += 0.07
                elif intent_lower == "symptoms" and any(k in cand_q for k in ["symptom", "sign", "feel"]):
                    intent_boost += 0.07
                elif intent_lower == "treatment" and any(k in cand_q for k in ["treat", "cure", "therap", "manage"]):
                    intent_boost += 0.07
                elif intent_lower == "prevention" and any(k in cand_q for k in ["prevent", "avoid"]):
                    intent_boost += 0.07
                elif intent_lower == "diagnosis" and any(k in cand_q for k in ["diagnos", "test", "screen"]):
                    intent_boost += 0.07

            # Entity token overlap boost
            overlap_count = 0
            if entity_tokens:
                overlap_count = len(entity_tokens.intersection(cand_tokens))
            else:
                overlap_count = len(query_tokens.intersection(cand_tokens))

            entity_boost = min(0.12, overlap_count * 0.04)

            # Combined reranked score bounded [0, 0.99]
            final_score = min(0.99, max(0.05, base_sim * 0.80 + intent_boost + entity_boost))

            candidates.append({
                "question": item["question"],
                "similarity_score": round(float(final_score), 4),
                "answer_snippet": item["answer_snippet"],
                "source": item["source"],
                "source_organization": item.get("source_organization", f"NIH / {item['source']}"),
                "focus_area": item["focus_area"]
            })

        # Sort by similarity score descending
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]


if __name__ == "__main__":
    engine = SimilarityEngine()
    engine.load()
    q = "I have been thirsty all day and going to the bathroom constantly, what could be wrong?"
    results = engine.search(q, top_k=4)
    print(f"\nSearch results for '{q}':")
    for r in results:
        print(f"- ({r['similarity_score']*100:.1f}%) {r['question']} [{r['source_organization']}]")
