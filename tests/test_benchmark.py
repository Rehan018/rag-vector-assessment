import numpy as np

from src.benchmark import BenchmarkRunner
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
            elif "retry" in lower_text or "dead-letter" in lower_text:
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

        if (
            "retry" in lower_query
            or "background job failures" in lower_query
            or "dead-letter" in lower_query
        ):
            return np.array([0.0, 1.0, 0.0], dtype="float32")

        return np.array([0.0, 0.0, 1.0], dtype="float32")


def make_retriever():
    documents = [
        DocumentChunk(
            id="chunk_001",
            title="Peak Load Handling",
            text="The system handles peak load using queues and backpressure.",
            metadata={"topic": "scalability"},
        ),
        DocumentChunk(
            id="chunk_002",
            title="Retry Handling",
            text="Failed jobs use retry budgets and dead-letter queues.",
            metadata={"topic": "reliability"},
        ),
        DocumentChunk(
            id="chunk_003",
            title="Retrieval Evaluation",
            text="Retrieval quality is measured using top-k ranking and expected chunks.",
            metadata={"topic": "evaluation"},
        ),
    ]

    embedding_provider = FakeEmbeddingProvider()
    embeddings = embedding_provider.embed_texts([document.text for document in documents])

    store = FaissVectorStore(dimension=3)
    store.add_documents(documents, embeddings)

    return Retriever(
        embedding_provider=embedding_provider,
        vector_store=store,
        query_expander=MockQueryExpander(),
    )


def test_benchmark_runner_returns_cases_for_queries():
    retriever = make_retriever()
    runner = BenchmarkRunner(retriever=retriever, top_k=2)

    cases = runner.run(
        [
            "How does the system handle peak load?",
            "What happens when background jobs fail repeatedly?",
        ]
    )

    assert len(cases) == 2
    assert cases[0].query == "How does the system handle peak load?"
    assert cases[0].expanded_query is not None
    assert len(cases[0].strategy_a_results) == 2
    assert len(cases[0].strategy_b_results) == 2


def test_benchmark_runner_writes_markdown_report(tmp_path):
    retriever = make_retriever()
    runner = BenchmarkRunner(retriever=retriever, top_k=2)

    cases = runner.run(["How does the system handle peak load?"])

    output_path = tmp_path / "retrieval_benchmark.md"

    runner.write_markdown_report(
        cases=cases,
        output_path=str(output_path),
    )

    content = output_path.read_text(encoding="utf-8")

    assert "# Retrieval Benchmark: Strategy A vs Strategy B" in content
    assert "Strategy A: Raw Vector Search" in content
    assert "Strategy B: AI-Enhanced Retrieval" in content
    assert "Expanded query" in content

def test_expected_chunk_metric_is_computed():
    retriever = make_retriever()
    runner = BenchmarkRunner(retriever=retriever, top_k=2)

    cases = runner.run(["How does the system handle peak load?"])

    metric = cases[0].metric

    assert metric is not None
    assert metric.expected_chunk_id == "chunk_001"

    assert metric.strategy_a_rank is not None
    assert metric.strategy_b_rank is not None

    assert 1 <= metric.strategy_a_rank <= 2
    assert 1 <= metric.strategy_b_rank <= 2

    assert metric.strategy_a_score is not None
    assert metric.strategy_b_score is not None
    assert metric.score_delta is not None

    assert metric.verdict in {
        "Same rank, stronger score after expansion",
        "No measurable change",
        "Same rank, weaker score after expansion",
        "Improved: expected chunk moved higher",
        "Worse: expected chunk moved lower",
    }