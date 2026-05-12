from src.config import config
from src.data_loader import DataLoader
from src.embeddings import SentenceTransformerEmbeddingProvider


def main() -> None:
    loader = DataLoader(config.corpus_path)
    chunks = loader.load()

    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name=config.embedding_model_name
    )

    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_provider.embed_texts(texts)

    query = "How does the system handle peak load?"
    query_embedding = embedding_provider.embed_query(query)

    print(f"Loaded {len(chunks)} document chunks")
    print(f"Document embedding shape: {embeddings.shape}")
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")


if __name__ == "__main__":
    main()