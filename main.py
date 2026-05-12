from src.config import config
from src.data_loader import DataLoader
from src.embeddings import SentenceTransformerEmbeddingProvider
from src.query_expander import MockQueryExpander
from src.retriever import Retriever
from src.vector_store import FaissVectorStore


def build_retriever() -> Retriever:
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


def print_results(title, results):
    print(title)
    print("-" * len(title))

    for result in results:
        print(f"Rank {result.rank}")
        print(f"Title: {result.title}")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.chunk_id}")
        print()

    print()


def main() -> None:
    retriever = build_retriever()

    query = "How does the system handle peak load?"

    raw_results = retriever.retrieve_raw(
        query=query,
        top_k=config.top_k,
    )

    expanded_query, expanded_results = retriever.retrieve_with_expansion(
        query=query,
        top_k=config.top_k,
    )

    print(f"Original Query: {query}")
    print(f"Expanded Query: {expanded_query}")
    print()

    print_results("Strategy A: Raw Vector Search", raw_results)
    print_results("Strategy B: AI-Enhanced Retrieval", expanded_results)


if __name__ == "__main__":
    main()