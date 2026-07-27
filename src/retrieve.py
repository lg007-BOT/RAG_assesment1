"""Vector retrieval with an explicit low-confidence gate."""
from dataclasses import dataclass
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, MAX_RETRIEVAL_DISTANCE, TOP_K


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    document: str
    page: int
    section: str
    distance: float


def retrieve(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("Index not found. Run `python -m src.ingest` first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    if collection.count() == 0:
        return []
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    vector = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(
            chunk_id=chunk_id, text=text, document=metadata["document"],
            page=int(metadata["page"]), section=metadata.get("section", "Unspecified"),
            distance=float(distance),
        )
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def has_sufficient_evidence(chunks: List[RetrievedChunk]) -> bool:
    return bool(chunks) and chunks[0].distance <= MAX_RETRIEVAL_DISTANCE
