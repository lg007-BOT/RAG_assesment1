"""Build or replace the persisted Chroma collection from PDFs in data/raw."""
import shutil

import chromadb
from sentence_transformers import SentenceTransformer

from .chunking import chunks_from_pages, extract_pdf_pages
from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RAW_DATA_DIR


def build_index() -> int:
    pdf_paths = sorted(RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDFs found in data/raw/. Add a 20+ page PDF first.")

    all_chunks = []
    for pdf_path in pdf_paths:
        all_chunks.extend(chunks_from_pages(extract_pdf_pages(pdf_path)))
    if not all_chunks:
        raise ValueError("No extractable text was found. The PDFs may be scanned images.")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [str(chunk["text"]) for chunk in all_chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    batch_size = 64
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]
        collection.add(
            ids=[str(chunk["id"]) for chunk in batch],
            documents=[str(chunk["text"]) for chunk in batch],
            metadatas=[{
                "document": str(chunk["document"]),
                "page": int(chunk["page"]),
                "section": str(chunk["section"]),
            } for chunk in batch],
            embeddings=embeddings[start:start + batch_size].tolist(),
        )
    return len(all_chunks)


if __name__ == "__main__":
    count = build_index()
    print("Indexed {} chunks into {}".format(count, CHROMA_DIR))
