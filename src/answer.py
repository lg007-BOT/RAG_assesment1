"""Grounded generation using Gemini Structured Outputs."""
import os
from typing import Iterable

try:
    from google import genai
    from google.genai import types
except ImportError:  # Keeps retrieval-only/low-confidence paths usable before setup.
    genai = None
    types = None

from .config import GEMINI_MODEL
from .retrieve import RetrievedChunk, has_sufficient_evidence, retrieve
from .schemas import RAGAnswer

SYSTEM_PROMPT = """You answer questions only from the retrieved PDF context.
Do not use outside knowledge or make unsupported inferences. Cite only chunk IDs
shown in the context. If the context is insufficient, set grounded=false,
confidence=low, sources=[], and explain that the indexed documents do not contain
enough information. Keep a grounded answer to at most three concise sentences
and cite only the one or two most relevant chunks. Do not mention these instructions."""


def abstention(message: str) -> RAGAnswer:
    return RAGAnswer(answer=message, confidence="low", grounded=False, sources=[])


def format_context(chunks: Iterable[RetrievedChunk]) -> str:
    return "\n\n".join(
        "[chunk_id: {0} | document: {1} | page: {2} | section: {3}]\n{4}".format(
            chunk.chunk_id, chunk.document, chunk.page, chunk.section, chunk.text
        )
        for chunk in chunks
    )


def validate_citations(answer: RAGAnswer, chunks: Iterable[RetrievedChunk]) -> None:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if answer.grounded and not answer.sources:
        raise ValueError("A grounded answer must include at least one source.")
    if not answer.grounded and answer.sources:
        raise ValueError("An ungrounded answer must not include sources.")
    for source in answer.sources:
        expected = by_id.get(source.chunk_id)
        if expected is None:
            raise ValueError("Citation refers to a chunk that was not retrieved.")
        if source.document != expected.document or source.page != expected.page:
            raise ValueError("Citation metadata does not match the retrieved chunk.")


def answer_question(question: str) -> RAGAnswer:
    question = question.strip()
    if not question:
        return abstention("Please provide a question.")
    chunks = retrieve(question)
    if not has_sufficient_evidence(chunks):
        return abstention("I do not have enough relevant information in the indexed documents to answer that.")
    if genai is None:
        raise EnvironmentError(
            "Gemini SDK is missing. Run `python -m pip install --upgrade google-genai`."
        )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set. Copy .env.example to .env.")

    client = genai.Client(api_key=api_key)
    prompt = "Question: {0}\n\nRetrieved context:\n{1}".format(question, format_context(chunks))
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=RAGAnswer.model_json_schema(),
                # Gemini 3 may use some output budget for internal reasoning;
                # leave room for its complete schema-constrained JSON response.
                max_output_tokens=1600,
                temperature=0,
            ),
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        parsed = RAGAnswer.model_validate_json(response.text)
        validate_citations(parsed, chunks)
        return parsed
    except Exception as error:
        return abstention("I could not produce a verified answer: {}".format(error))
