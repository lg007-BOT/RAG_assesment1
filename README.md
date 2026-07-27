# Grounded RAG with Gemini Structured Output

This project indexes one or more PDFs, retrieves relevant passages with Chroma,
and returns a Gemini answer that is schema-validated and traceable to the exact
document page and chunk used.

## Quick start

1. Use Python 3.9+ and create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy the environment template and set your API key:

   ```bash
   cp .env.example .env
   ```

   If you created `.venv` before switching this project to Gemini, refresh the
   dependencies first:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Put at least one 20+ page PDF in `data/raw/`. The repository intentionally
   does not ship a document set, so you can use a public PDF you are permitted
   to distribute (for example, the NIST AI RMF PDF).

4. Build the local index:

   ```bash
   python -m src.ingest
   ```

5. Ask a question:

   ```bash
   python demo.py "What does the document say about governing AI risks?"
   ```

## Design highlights

- Sentence-aware chunks target 500 words with 100 words of overlap.
- Chroma uses normalized SentenceTransformer embeddings and cosine distance.
- Retrieval returns a safe abstention before an LLM call when every hit is weak.
- Gemini Structured Outputs + Pydantic enforce the response schema.
- Returned citations are checked against the chunks actually retrieved, which
  prevents unsupported citation IDs from being presented as evidence.

See `writeup.md` for submission-ready design notes and `loom_script.md` for the
video walkthrough.
