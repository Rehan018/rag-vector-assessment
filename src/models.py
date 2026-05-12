from pydantic import BaseModel, Field
from typing import List, Optional


class DocumentChunk(BaseModel):
    id: str = Field(..., description="Stable unique ID for the chunk")
    title: str = Field(..., description="Short title describing the chunk")
    text: str = Field(..., description="Chunk content used for embedding and retrieval")
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    title: str
    text: str
    score: float
    rank: int
    strategy: str


class BenchmarkCase(BaseModel):
    query: str
    expanded_query: Optional[str] = None
    strategy_a_results: List[RetrievalResult]
    strategy_b_results: List[RetrievalResult]