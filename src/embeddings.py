from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    """
    Interface for embedding providers.

    The retrieval system depends on this abstraction instead of depending
    directly on sentence-transformers or Vertex AI. This keeps the local
    assignment implementation easy to migrate to a managed embedding service.
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        pass


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    This simulates the behavior of a managed embedding model such as
    Vertex AI textembedding-gecko while keeping the assessment fully local.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts cannot be empty")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype("float32")[0]


class MockVertexTextEmbeddingModel:
    """
    Lightweight mock of vertexai.language_models.TextEmbeddingModel.

    This is intentionally simple. The purpose is to show how the code would
    interact with a Vertex-style embedding client without requiring GCP access
    during local assessment or tests.
    """

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    @classmethod
    def from_pretrained(cls, model_name: str):
        provider = SentenceTransformerEmbeddingProvider(model_name)
        return cls(provider)

    def get_embeddings(self, texts: List[str]) -> List[dict]:
        vectors = self.embedding_provider.embed_texts(texts)

        return [
            {
                "values": vector.tolist()
            }
            for vector in vectors
        ]