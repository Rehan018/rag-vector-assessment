from typing import List, Optional

from src.embeddings import EmbeddingProvider
from src.models import RetrievalResult
from src.query_expander import QueryExpander
from src.vector_store import FaissVectorStore


class Retriever:
    """
    Coordinates retrieval strategies over a vector store.

    Strategy A:
        Raw user query -> embedding -> vector search

    Strategy B:
        User query -> query expansion -> embedding -> vector search
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: FaissVectorStore,
        query_expander: Optional[QueryExpander] = None,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.query_expander = query_expander

    def retrieve_raw(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        query_embedding = self.embedding_provider.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return self._to_retrieval_results(
            results=results,
            strategy="raw_vector_search",
        )

    def retrieve_with_expansion(
        self,
        query: str,
        top_k: int,
    ) -> tuple[str, List[RetrievalResult]]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if self.query_expander is None:
            raise ValueError("query_expander is required for expanded retrieval")

        expanded_query = self.query_expander.expand(query)
        expanded_embedding = self.embedding_provider.embed_query(expanded_query)

        results = self.vector_store.search(
            query_embedding=expanded_embedding,
            top_k=top_k,
        )

        retrieval_results = self._to_retrieval_results(
            results=results,
            strategy="ai_enhanced_retrieval",
        )

        return expanded_query, retrieval_results

    def _to_retrieval_results(
        self,
        results,
        strategy: str,
    ) -> List[RetrievalResult]:
        retrieval_results: List[RetrievalResult] = []

        for rank, (document, score) in enumerate(results, start=1):
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=document.id,
                    title=document.title,
                    text=document.text,
                    score=score,
                    rank=rank,
                    strategy=strategy,
                )
            )

        return retrieval_results