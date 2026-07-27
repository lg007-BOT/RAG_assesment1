"""The response contract returned by the RAG system."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    document: str = Field(description="Source PDF filename supplied in the retrieved context.")
    page: int = Field(description="One-indexed page number in the source PDF.")
    section: Optional[str] = Field(default=None, description="Detected section heading, if available.")
    chunk_id: str = Field(description="Identifier of a retrieved context chunk.")


class RAGAnswer(BaseModel):
    answer: str = Field(description="A concise answer supported only by the provided context.")
    confidence: Literal["high", "medium", "low"]
    grounded: bool = Field(description="True only when the answer is supported by supplied context.")
    sources: List[Source] = Field(description="Only chunks that support the answer.")
