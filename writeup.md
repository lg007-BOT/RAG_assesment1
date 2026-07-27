# RAG Pipeline with Structured Output — Write-up

## Document set

I indexed a focused set of PDF documents stored in `data/raw/` (at least 20
pages total). Each extracted chunk retains its original document filename, page
number, detected section heading when available, and a unique chunk ID. This
makes an answer auditable against the original PDF rather than merely claiming
that it came from the corpus.

## Chunking and embedding

The pipeline extracts text page by page and uses sentence-aware chunks targeted
at 500 words with a 100-word overlap. This size is large enough to retain a
complete policy statement or explanation, but small enough that semantic search
returns focused evidence. The overlap preserves context when a key statement
falls at the boundary between chunks. Chunks are embedded with the local
`all-MiniLM-L6-v2` SentenceTransformer model and persisted in Chroma using
cosine distance.

## Retrieval and grounding

For a question, the system retrieves the top four chunks. It applies a
low-confidence gate before calling the LLM: if the best cosine distance exceeds
the configured threshold (default 0.80), the application returns a structured
abstention. This prevents an irrelevant question from becoming a plausible but
unsupported answer. The threshold is intentionally configurable and should be
tuned using the included evaluation questions for the selected corpus.

## Schema design

Gemini is called using Structured Outputs and a Pydantic `RAGAnswer` schema.
Every response contains `answer`, constrained `confidence`, `grounded`, and a
list of sources (`document`, `page`, `section`, and `chunk_id`). The generation
prompt restricts Gemini to retrieved context. After Gemini responds, the code
checks every cited chunk ID and its page/document metadata against the actual
retrieval results. Invalid citations, missing output, API failures, and schema
errors all produce a safe low-confidence structured abstention rather than an
unverified answer.
