import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "384"))
    top_k: int = int(os.getenv("TOP_K", "3"))

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "https://ollama.merai.app")
    ollama_model_name: str = os.getenv("OLLAMA_MODEL_NAME", "llama4:latest")

    corpus_path: str = os.getenv("CORPUS_PATH", "data/technical_corpus.json")
    benchmark_output_path: str = os.getenv("BENCHMARK_OUTPUT_PATH", "retrieval_benchmark.md")


config = AppConfig()