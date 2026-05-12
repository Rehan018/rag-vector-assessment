from pathlib import Path
from typing import List

from tabulate import tabulate

from src.config import config
from src.data_loader import DataLoader
from src.embeddings import SentenceTransformerEmbeddingProvider
from src.models import BenchmarkCase, RetrievalResult
from src.query_expander import MockQueryExpander
from src.retriever import Retriever
from src.vector_store import FaissVectorStore


BENCHMARK_QUERIES = [
    "How does the system handle peak load?",
    "What happens when background jobs fail repeatedly?",
    "How can we know if retrieval quality is getting worse?",
]


class BenchmarkRunner:
    """
    Runs Strategy A vs Strategy B retrieval comparisons.

    Strategy A uses the raw user query.
    Strategy B expands the query before embedding and retrieval.
    """

    def __init__(self, retriever: Retriever, top_k: int):
        self.retriever = retriever
        self.top_k = top_k

    def run(self, queries: List[str]) -> List[BenchmarkCase]:
        cases: List[BenchmarkCase] = []

        for query in queries:
            raw_results = self.retriever.retrieve_raw(
                query=query,
                top_k=self.top_k,
            )

            expanded_query, expanded_results = self.retriever.retrieve_with_expansion(
                query=query,
                top_k=self.top_k,
            )

            cases.append(
                BenchmarkCase(
                    query=query,
                    expanded_query=expanded_query,
                    strategy_a_results=raw_results,
                    strategy_b_results=expanded_results,
                )
            )

        return cases

    def write_markdown_report(
        self,
        cases: List[BenchmarkCase],
        output_path: str,
    ) -> None:
        path = Path(output_path)

        lines = [
            "# Retrieval Benchmark: Strategy A vs Strategy B",
            "",
            "This benchmark compares two retrieval strategies over the same local technical corpus.",
            "",
            "- Strategy A: raw vector search using the original user query.",
            "- Strategy B: AI-enhanced retrieval using query expansion before embedding.",
            "",
            "Embeddings are normalized, and FAISS inner-product search is used as cosine-style similarity.",
            "",
        ]

        for index, case in enumerate(cases, start=1):
            lines.extend(
                [
                    f"## Query {index}",
                    "",
                    f"Original query: `{case.query}`",
                    "",
                    f"Expanded query: `{case.expanded_query}`",
                    "",
                    "### Strategy A: Raw Vector Search",
                    "",
                    self._format_results_table(case.strategy_a_results),
                    "",
                    "### Strategy B: AI-Enhanced Retrieval",
                    "",
                    self._format_results_table(case.strategy_b_results),
                    "",
                    "### Observation",
                    "",
                    self._build_observation(case),
                    "",
                ]
            )

        path.write_text("\n".join(lines), encoding="utf-8")

    def _format_results_table(self, results: List[RetrievalResult]) -> str:
        table = [
            [
                result.rank,
                result.chunk_id,
                result.title,
                f"{result.score:.4f}",
            ]
            for result in results
        ]

        return tabulate(
            table,
            headers=["Rank", "Chunk ID", "Title", "Score"],
            tablefmt="github",
        )

    def _build_observation(self, case: BenchmarkCase) -> str:
        raw_top = case.strategy_a_results[0].chunk_id
        expanded_top = case.strategy_b_results[0].chunk_id

        if raw_top == expanded_top:
            return (
                "Both strategies retrieved the same top-ranked chunk. "
                "The expanded query may still help by changing score distribution "
                "or improving recall in lower-ranked results."
            )

        return (
            "The expanded query changed the top-ranked result. "
            "This indicates that query rewriting can materially affect retrieval behavior, "
            "which should be evaluated carefully to avoid query drift."
        )


def build_default_retriever() -> Retriever:
    loader = DataLoader(config.corpus_path)
    chunks = loader.load()

    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name=config.embedding_model_name
    )

    document_embeddings = embedding_provider.embed_texts(
        [chunk.text for chunk in chunks]
    )

    vector_store = FaissVectorStore(dimension=config.vector_dimension)
    vector_store.add_documents(chunks, document_embeddings)

    return Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        query_expander=MockQueryExpander(),
    )


def main() -> None:
    retriever = build_default_retriever()

    runner = BenchmarkRunner(
        retriever=retriever,
        top_k=config.top_k,
    )

    cases = runner.run(BENCHMARK_QUERIES)

    runner.write_markdown_report(
        cases=cases,
        output_path=config.benchmark_output_path,
    )

    print(f"Benchmark written to {config.benchmark_output_path}")


if __name__ == "__main__":
    main()