from src.benchmark import BENCHMARK_QUERIES, BenchmarkRunner, build_default_retriever
from src.config import config


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

    print("RAG retrieval benchmark completed successfully.")
    print(f"Report written to: {config.benchmark_output_path}")
    print()
    print("Compared strategies:")
    print("- Strategy A: raw vector search")
    print("- Strategy B: query expansion + vector search")
    print()
    print(f"Queries evaluated: {len(BENCHMARK_QUERIES)}")
    print(f"Top-k per query: {config.top_k}")


if __name__ == "__main__":
    main()