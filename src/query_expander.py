from abc import ABC, abstractmethod
from typing import Dict

import requests


class QueryExpansionError(Exception):
    pass


class QueryExpander(ABC):
    """
    Interface for query expansion.

    Retrieval code depends on this abstraction instead of depending directly
    on Ollama, Vertex AI, or any specific LLM provider.
    """

    @abstractmethod
    def expand(self, query: str) -> str:
        pass


class MockQueryExpander(QueryExpander):
    """
    Deterministic query expander used for tests and reproducible benchmarks.

    This avoids flaky test behavior caused by non-deterministic LLM output.
    """

    def __init__(self):
        self.expansions: Dict[str, str] = {
            "How does the system handle peak load?": (
                "How does the system handle peak load, traffic spikes, "
                "queue depth, horizontal worker scaling, stateless API servers, "
                "backpressure, and processing latency?"
            ),
            "What happens when background jobs fail repeatedly?": (
                "What retry strategy is used for repeated background job failures, "
                "including exponential backoff, jitter, retry budget, and dead-letter queues?"
            ),
            "How can we know if retrieval quality is getting worse?": (
                "How is retrieval quality evaluated using top-k results, expected chunk rank, "
                "similarity score, benchmark queries, and whether query expansion improves relevance?"
            ),
        }

    def expand(self, query: str) -> str:
        if not query or not query.strip():
            raise QueryExpansionError("query cannot be empty")

        return self.expansions.get(
            query,
            f"{query} Include related technical terms, failure modes, scalability concerns, and operational signals.",
        )


class OllamaQueryExpander(QueryExpander):
    """
    Local LLM-backed query expander using Ollama.

    This keeps the project local while still demonstrating how an AI-enhanced
    retrieval strategy can rewrite user queries before embedding.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def expand(self, query: str) -> str:
        if not query or not query.strip():
            raise QueryExpansionError("query cannot be empty")

        prompt = self._build_prompt(query)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    },
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise QueryExpansionError(f"Ollama query expansion failed: {exc}") from exc

        data = response.json()
        expanded_query = data.get("response", "").strip()

        if not expanded_query:
            raise QueryExpansionError("Ollama returned an empty expanded query")

        return expanded_query

    def _build_prompt(self, query: str) -> str:
        return f"""
You are improving a search query for a technical RAG retrieval system.

Rewrite the user query into a clearer embedding-friendly query.
Add relevant technical terms, but do not answer the question.
Do not add unrelated assumptions.
Return only the rewritten query.

User query:
{query}
""".strip()


class MockVertexGenerativeModel:
    """
    Lightweight mock of vertexai.language_models.GenerativeModel.

    The assignment asks for mocking the GenerativeModel used in query expansion.
    This class gives us a Vertex-style interface while keeping execution local.
    """

    def __init__(self, query_expander: QueryExpander):
        self.query_expander = query_expander

    def generate_content(self, prompt: str):
        expanded = self.query_expander.expand(prompt)

        return {
            "text": expanded
        }