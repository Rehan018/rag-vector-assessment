import pytest

from src.query_expander import (
    MockQueryExpander,
    MockVertexGenerativeModel,
    QueryExpansionError,
)


def test_mock_query_expander_returns_deterministic_expansion():
    expander = MockQueryExpander()

    expanded = expander.expand("How does the system handle peak load?")

    assert "traffic spikes" in expanded
    assert "backpressure" in expanded
    assert "worker scaling" in expanded


def test_mock_query_expander_fallback_for_unknown_query():
    expander = MockQueryExpander()

    expanded = expander.expand("How is caching handled?")

    assert "How is caching handled?" in expanded
    assert "technical terms" in expanded


def test_mock_query_expander_rejects_empty_query():
    expander = MockQueryExpander()

    with pytest.raises(QueryExpansionError):
        expander.expand("   ")


def test_mock_vertex_generative_model_returns_vertex_like_response():
    expander = MockQueryExpander()
    model = MockVertexGenerativeModel(expander)

    response = model.generate_content("How does the system handle peak load?")

    assert hasattr(response, "text")
    assert "backpressure" in response.text