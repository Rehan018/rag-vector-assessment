import numpy as np
import pytest

from src.embeddings import EmbeddingProvider
from src.models import DocumentChunk
from src.query_expander import MockQueryExpander
from src.retriever import Retriever
from src.vector_store import FaissVectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts):
        vectors = []

        for text in texts:
            lower_text = text.lower()

            if "peak load" in lower_text or "queue" in lower_text:
                vectors.append([1.0, 0.0, 0.0])
            elif "authentication" in lower_text or "token" in lower_text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return np.array(vectors, dtype="float32")

    def embed_query(self, query):
        lower_query = query.lower()

        if (
            "peak load" in lower_query
            or "traffic spikes" in lower_query
            or "backpressure" in lower_query
        ):
            return np.array([1.0, 0.0, 0.0], dtype="float32")

        if "authentication" in lower_query or "token" in lower_query:
            return np.array([0.0, 1.0, 0.0], dtype="float32")

        return np.array([0.0, 0.0, 1.0], dtype="float32")


def make_store():
    documents = [
        DocumentChunk(
            id="chunk_001",
            title="Peak Load Handling",
            text="The system handles peak load using queues, workers, and backpressure.",
            metadata={"topic": "scalability"},
        ),
        DocumentChunk(
            id="chunk_002",
            title="Authentication",
            text="The system authenticates users using signed tokens.",
            metadata={"topic": "security"},
        ),
        DocumentChunk(
            id="chunk_003",
            title="Observability",
            text="The system tracks logs, metrics, and alerts.",
            metadata={"topic": "observability"},
        ),
    ]

    embedding_provider = FakeEmbeddingProvider()
    embeddings = embedding_provider.embed_texts([doc.text for doc in documents])

    store = FaissVectorStore(dimension=3)
    store.add_documents(documents, embeddings)

    return store, embedding_provider


def test_retrieve_raw_returns_ranked_results():
    store, embedding_provider = make_store()

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=store,
    )

    results = retriever.retrieve_raw(
        query="How does the system handle peak load?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk_001"
    assert results[0].rank == 1
    assert results[0].strategy == "raw_vector_search"


def test_retrieve_with_expansion_returns_expanded_query_and_results():
    store, embedding_provider = make_store()

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=store,
        query_expander=MockQueryExpander(),
    )

    expanded_query, results = retriever.retrieve_with_expansion(
        query="How does the system handle peak load?",
        top_k=2,
    )

    assert "traffic spikes" in expanded_query
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_001"
    assert results[0].strategy == "ai_enhanced_retrieval"


def test_expanded_retrieval_requires_query_expander():
    store, embedding_provider = make_store()

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=store,
    )

    with pytest.raises(ValueError):
        retriever.retrieve_with_expansion(
            query="How does the system handle peak load?",
            top_k=2,
        )


def test_raw_retrieval_rejects_empty_query():
    store, embedding_provider = make_store()

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=store,
    )

    with pytest.raises(ValueError):
        retriever.retrieve_raw(query=" ", top_k=2)