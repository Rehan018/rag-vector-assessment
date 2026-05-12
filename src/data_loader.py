import json
from pathlib import Path
from typing import List

from src.models import DocumentChunk


class CorpusLoadError(Exception):
    pass


class DataLoader:
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)

    def load(self) -> List[DocumentChunk]:
        if not self.corpus_path.exists():
            raise CorpusLoadError(f"Corpus file not found: {self.corpus_path}")

        try:
            with self.corpus_path.open("r", encoding="utf-8") as file:
                raw_items = json.load(file)
        except json.JSONDecodeError as exc:
            raise CorpusLoadError(f"Invalid JSON corpus: {exc}") from exc

        if not isinstance(raw_items, list):
            raise CorpusLoadError("Corpus must be a list of document chunks")

        chunks: List[DocumentChunk] = []

        for item in raw_items:
            chunks.append(DocumentChunk(**item))

        if not chunks:
            raise CorpusLoadError("Corpus is empty")

        return chunks