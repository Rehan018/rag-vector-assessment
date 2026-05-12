import numpy as np
import pytest

from src.embeddings import EmbeddingProvider, MockVertexTextEmbeddingModel


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype="float32",
        )

    def embed_query(self, query):
        return np.array([1.0, 0.0, 0.0], dtype="float32")


def test_mock_vertex_text_embedding_model_returns_vertex_like_shape():
    provider = FakeEmbeddingProvider()
    model = MockVertexTextEmbeddingModel(provider)

    result = model.get_embeddings(["first text", "second text"])

    assert len(result) == 2
    assert "values" in result[0]
    assert result[0]["values"] == [1.0, 0.0, 0.0]
    assert result[1]["values"] == [0.0, 1.0, 0.0]


def test_fake_embedding_provider_query_shape():
    provider = FakeEmbeddingProvider()

    vector = provider.embed_query("peak load")

    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert vector.shape == (3,)