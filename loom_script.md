# Loom walkthrough (6–8 minutes)

1. **Goal (0:00–0:40):** Explain that this is a RAG application whose output is
   structured and traceable, not ordinary chatbot text. Identify the PDF set.
2. **Ingestion (0:40–2:10):** Show `data/raw/`, then run `python -m src.ingest`.
   Explain 500-word sentence-aware chunks, 100-word overlap, and page metadata.
3. **Retrieval (2:10–3:20):** Open `src/retrieve.py`; explain top-4 cosine search
   and the low-confidence distance gate.
4. **Structured generation (3:20–4:30):** Open `src/schemas.py` and
   `src/answer.py`. Show the Pydantic schema, Gemini `generate_content` with
   Structured Outputs, and the
   post-generation citation validation.
5. **Live success demos (4:30–6:10):** Ask two questions explicitly answered in
   your PDF. Point out `grounded: true`, source page numbers, and chunk IDs.
6. **Failure demo (6:10–7:10):** Ask an unrelated question such as “Who won the
   2022 FIFA World Cup?” Show the low-confidence structured abstention.
7. **Close (7:10–7:30):** Mention the evaluation fixture and explain that the
   threshold is tuned to the selected corpus.
