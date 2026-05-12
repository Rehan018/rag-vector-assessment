import numpy as np
import pytest

from src.models import DocumentChunk
from src.vector_store import FaissVectorStore, VectorStoreError


def make_documents():
    return [
        DocumentChunk(
            id="chunk_001",
            title="Peak Load Handling",
            text="The system handles peak load using queues and workers.",
            metadata={"topic": "scalability"},
        ),
        DocumentChunk(
            id="chunk_002",
            title="Authentication",
            text="The system authenticates users with signed tokens.",
            metadata={"topic": "security"},
        ),
    ]


def test_add_documents_and_count():
    store = FaissVectorStore(dimension=3)
    docs = make_documents()

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )

    store.add_documents(docs, embeddings)

    assert store.count() == 2


def test_search_returns_most_similar_document():
    store = FaissVectorStore(dimension=3)
    docs = make_documents()

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )

    store.add_documents(docs, embeddings)

    query_embedding = np.array([1.0, 0.0, 0.0], dtype="float32")
    results = store.search(query_embedding=query_embedding, top_k=1)

    assert len(results) == 1

    document, score = results[0]

    assert document.id == "chunk_001"
    assert score == pytest.approx(1.0)


def test_search_empty_store_raises_error():
    store = FaissVectorStore(dimension=3)

    query_embedding = np.array([1.0, 0.0, 0.0], dtype="float32")

    with pytest.raises(VectorStoreError):
        store.search(query_embedding=query_embedding, top_k=1)


def test_dimension_mismatch_raises_error():
    store = FaissVectorStore(dimension=3)
    docs = make_documents()

    wrong_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype="float32",
    )

    with pytest.raises(VectorStoreError):
        store.add_documents(docs, wrong_embeddings)