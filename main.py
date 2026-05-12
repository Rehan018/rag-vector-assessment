from src.config import config
from src.data_loader import DataLoader
from src.embeddings import SentenceTransformerEmbeddingProvider
from src.vector_store import FaissVectorStore


def main() -> None:
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

    query = "How does the system handle peak load?"
    query_embedding = embedding_provider.embed_query(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=config.top_k,
    )

    print(f"Query: {query}")
    print()

    for rank, (document, score) in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"Title: {document.title}")
        print(f"Score: {score:.4f}")
        print(f"Text: {document.text}")
        print()


if __name__ == "__main__":
    main()