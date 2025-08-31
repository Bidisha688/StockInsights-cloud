from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import os

import numpy as np
from sentence_transformers import SentenceTransformer

# Optional deps (import lazily/safely)
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    import chromadb  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore

try:
    import weaviate  # type: ignore
except Exception:  # pragma: no cover
    weaviate = None  # type: ignore

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


# -------------------- Base Interface --------------------
class VectorStoreBase:
    def build(self, chunks: List[Tuple[str, Dict]]):
        raise NotImplementedError

    def search(self, query: str, k: int = 5):
        raise NotImplementedError

    def save(self, path: str):  # optional
        pass

    def load(self, path: str):  # optional
        pass


# -------------------- FAISS Backend --------------------
class FaissVectorStore(VectorStoreBase):
    """In-process FAISS (default). Cosine similarity via normalized vectors."""
    def __init__(self):
        if faiss is None:
            raise RuntimeError("faiss-cpu not installed; install or choose another backend.")
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index: Optional["faiss.IndexFlatIP"] = None  # type: ignore
        self.meta: List[Dict] = []

    def build(self, chunks: List[Tuple[str, Dict]]):
        if not chunks:
            self.index, self.meta = None, []
            return
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
        self.index.add(emb)
        self.meta = metas

    def search(self, query: str, k: int = 5):
        if self.index is None or not self.meta:
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            results.append((self.meta[idx], float(score)))
        return results

    def save(self, path: str):
        if self.index is None or not self.meta:
            return
        faiss.write_index(self.index, path + ".faiss")
        import json
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f)

    def load(self, path: str):
        try:
            self.index = faiss.read_index(path + ".faiss")
            import json
            with open(path + ".meta.json", "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        except Exception:
            self.index, self.meta = None, []


# -------------------- Chroma Backend --------------------
class ChromaVectorStore(VectorStoreBase):
    """Local persistent vector store using ChromaDB. Uses our own embeddings."""
    def __init__(self, collection: str = "news_chunks", persist_path: Optional[str] = None):
        if chromadb is None:
            raise RuntimeError("chromadb not installed; pip install chromadb or choose another backend.")
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.persist_path = persist_path or os.getenv("CHROMA_PATH")
        self.client = (
            chromadb.PersistentClient(path=self.persist_path)
            if self.persist_path else chromadb.Client()
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.ids: List[str] = []
        self.meta: List[Dict] = []

    def build(self, chunks: List[Tuple[str, Dict]]):
        # Reset collection to avoid stale docs when re-indexing
        if self.ids:
            try:
                self.collection.delete(ids=self.ids)
            except Exception:
                pass
        if not chunks:
            self.ids, self.meta = [], []
            return
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        ids = [f"doc-{i}" for i in range(len(texts))]
        self.collection.upsert(ids=ids, embeddings=emb.tolist(), metadatas=metas, documents=texts)
        self.ids, self.meta = ids, metas

    def search(self, query: str, k: int = 5):
        if not self.meta:
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        res = self.collection.query(query_embeddings=q.tolist(), n_results=k)
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        results: List[Tuple[Dict, float]] = []
        for m, d in zip(metadatas, distances):
            # Convert cosine distance → similarity score
            score = 1.0 - float(d)
            results.append((m, score))
        return results


# -------------------- Weaviate Backend --------------------
class WeaviateVectorStore(VectorStoreBase):
    """Remote vector DB via Weaviate (requires WEAVIATE_URL and optional WEAVIATE_API_KEY).
    NOTE: This implementation expects a pre-created class with vectorizer="none" and HNSW index.
    Set WEAVIATE_CLASS (default=NewsChunk).
    """
    def __init__(self, class_name: Optional[str] = None):
        if weaviate is None:
            raise RuntimeError("weaviate-client not installed; pip install weaviate-client or choose another backend.")
        url = os.getenv("WEAVIATE_URL")
        if not url:
            raise RuntimeError("Set WEAVIATE_URL to use Weaviate backend.")
        api_key = os.getenv("WEAVIATE_API_KEY")
        auth = weaviate.AuthApiKey(api_key=api_key) if api_key else None  # type: ignore
        self.client = weaviate.Client(url=url, auth_client_secret=auth)  # type: ignore
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.class_name = (class_name or os.getenv("WEAVIATE_CLASS") or "NewsChunk")
        self.meta: List[Dict] = []

    def build(self, chunks: List[Tuple[str, Dict]]):
        if not chunks:
            self.meta = []
            return
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        # Upsert objects with explicit vectors (vectorizer="none")
        with self.client.batch() as batch:  # type: ignore
            for i, (vec, meta) in enumerate(zip(emb, metas)):
                props = {k: v for k, v in meta.items() if k in ("title", "url", "source", "publishedAt", "text")}
                props.setdefault("text", texts[i])
                batch.add_data_object(
                    data_object=props,
                    class_name=self.class_name,
                    vector=vec.astype(np.float32),
                )
        self.meta = metas

    def search(self, query: str, k: int = 5):
        if not self.meta:
            return []
        q_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

        # Build and execute query
        res = (
            self.client.query
            .get(self.class_name, ["title", "url", "source", "publishedAt", "text"])  # type: ignore
            .with_near_vector({"vector": q_vec})  # type: ignore
            .with_additional(["distance"])  # type: ignore
            .with_limit(k)
            .do()
        )

        # Robust, step-by-step parsing (avoids giant parenthesis chains)
        data_block = res or {}
        get_block = data_block.get("data") or {}
        class_block = get_block.get("Get") or {}
        hits = class_block.get(self.class_name, []) or []

        out: List[Tuple[Dict, float]] = []
        for h in hits:
            meta = {k: h.get(k) for k in ("title", "url", "source", "publishedAt", "text")}
            dist = float(((h.get("_additional") or {}).get("distance") or 0.0))
            score = 1.0 - dist  # distance → similarity-like score
            out.append((meta, score))
        return out


# -------------------- Factory --------------------
def get_vector_store(backend: Optional[str] = None) -> VectorStoreBase:
    b = (backend or os.getenv("VECTOR_BACKEND") or "faiss").strip().lower()
    if b == "chroma":
        return ChromaVectorStore()
    if b == "weaviate":
        return WeaviateVectorStore()
    return FaissVectorStore()

# Back-compat alias so older imports still work (defaults to FAISS)
VectorStore = FaissVectorStore
