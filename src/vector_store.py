from typing import List, Tuple

import faiss
import numpy as np

from src.models import DocumentChunk


class VectorStoreError(Exception):
    pass


class FaissVectorStore:
    """
    Lightweight FAISS-backed vector store.

    Uses IndexFlatIP because embeddings are normalized before indexing.
    With normalized vectors, inner product is equivalent to cosine similarity.
    """

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[DocumentChunk] = []

    def add_documents(
        self,
        documents: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        if not documents:
            raise VectorStoreError("documents cannot be empty")

        if embeddings.ndim != 2:
            raise VectorStoreError("embeddings must be a 2D array")

        if embeddings.shape[0] != len(documents):
            raise VectorStoreError(
                "number of embeddings must match number of documents"
            )

        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Tuple[DocumentChunk, float]]:
        if len(self.documents) == 0:
            raise VectorStoreError("cannot search an empty vector store")

        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than zero")

        if query_embedding.ndim != 1:
            raise VectorStoreError("query_embedding must be a 1D vector")

        if query_embedding.shape[0] != self.dimension:
            raise VectorStoreError(
                f"query dimension mismatch: expected {self.dimension}, "
                f"got {query_embedding.shape[0]}"
            )

        query = query_embedding.astype("float32").reshape(1, -1)

        scores, indices = self.index.search(query, top_k)

        results: List[Tuple[DocumentChunk, float]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append((self.documents[idx], float(score)))

        return results

    def count(self) -> int:
        return len(self.documents)