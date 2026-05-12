from src.config import config
from src.data_loader import DataLoader


def main() -> None:
    loader = DataLoader(config.corpus_path)
    chunks = loader.load()

    print(f"Loaded {len(chunks)} document chunks")
    for chunk in chunks:
        print(f"- {chunk.id}: {chunk.title}")


if __name__ == "__main__":
    main()