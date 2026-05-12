# Context-Aware Retrieval Engine

This project is a local Retrieval-Augmented Generation retrieval pipeline built for the Gen AI assessment.

The goal is not to build a full chatbot. The main objective is to compare two retrieval strategies over the same technical corpus:

- Strategy A: raw vector search using the original user query
- Strategy B: query expansion followed by vector search

The project uses local embeddings, FAISS for vector search, and a deterministic query expansion mock for reproducible benchmarking. An Ollama-backed query expander is also included to show how the same interface can be connected to a local LLM.

## Problem Understanding

The assignment focuses on retrieval quality, not answer generation.

A normal RAG system usually has many moving parts: chunking, embeddings, vector search, prompt construction, generation, evaluation, monitoring, and production indexing. For this assessment, I kept the scope intentionally focused on the retrieval layer because that is what the prompt is testing.

The key question I tried to answer is:

> Does rewriting a user query before embedding change the quality or behavior of retrieved chunks?

To make that visible, the benchmark compares the top-3 retrieved chunks from raw vector search against the top-3 chunks retrieved after query expansion.

## Architecture

The project is split into small modules so each part has a clear responsibility.

```text
rag-vector-assessment/
  src/
    config.py              # shared configuration
    data_loader.py         # loads the local technical corpus
    embeddings.py          # embedding provider abstraction and local implementation
    vector_store.py        # FAISS-backed vector store
    query_expander.py      # mock and Ollama-based query expansion
    retriever.py           # Strategy A and Strategy B retrieval logic
    benchmark.py           # benchmark runner and markdown report writer
    run_demo.py            # evaluator-friendly entry point
    models.py              # typed data models
  data/
    technical_corpus.json  # small technical corpus used for indexing
  tests/
    test_embeddings.py
    test_vector_store.py
    test_query_expander.py
    test_retriever.py
    test_benchmark.py
    test_run_demo.py
  retrieval_benchmark.md
  main.py
  requirements.txt
  pyproject.toml
```

The main flow is:

```
Load corpus
   ↓
Generate embeddings
   ↓
Store vectors in FAISS
   ↓
Run raw query retrieval
   ↓
Run expanded query retrieval
   ↓
Write benchmark comparison
```

## Design Decisions

### 1. Embedding provider abstraction

The retrieval code does not depend directly on sentence-transformers.

Instead, it uses an EmbeddingProvider interface. The current implementation uses sentence-transformers/all-MiniLM-L6-v2, but the rest of the system does not care where the embedding comes from.

I did this because the assignment asks for a local setup that simulates Vertex AI embeddings. In a production setup, this boundary makes it easier to replace the local model with Vertex AI textembedding-gecko or another managed embedding model.

### 2. FAISS as the local vector store

I used FAISS because it is lightweight, fast, and easy to run locally. For this assessment, a managed vector database would be unnecessary because the corpus is small and the goal is to demonstrate retrieval logic, not infrastructure setup.

The vector store is wrapped in a FaissVectorStore class instead of being used directly throughout the code. This keeps FAISS-specific logic contained.

### 3. Query expansion is behind an interface

There are two query expansion implementations:

- MockQueryExpander
- OllamaQueryExpander

The mock expander is used for tests and benchmark reproducibility. LLM output can vary, so using a live LLM in tests would make the test suite flaky.

The Ollama implementation is included to show how the system can use a local LLM for query rewriting without changing the retriever logic.

### 4. Benchmark over at least three complex queries

The benchmark uses three queries that cover different retrieval situations:

- peak load handling
- repeated background job failures
- retrieval quality degradation

For each query, the system writes both raw and expanded retrieval results into retrieval_benchmark.md.

## Similarity Metric

The project uses normalized embeddings and FAISS inner-product search.

Because the embeddings are normalized before indexing, inner product behaves like cosine similarity. This is useful for semantic search because cosine similarity focuses on the direction of vectors rather than their raw magnitude.

I chose this approach because semantic text embeddings usually care more about meaning similarity than vector length. It also keeps the FAISS setup simple while still matching the behavior expected from cosine-based retrieval.

## Running the Project

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the benchmark:

```bash
python main.py
```

Or run the module directly:

```bash
python -m src.run_demo
```

This generates or updates:

- `retrieval_benchmark.md`

## Running Tests

```bash
pytest
```

The tests use fake embeddings and deterministic mocks where possible. This keeps the test suite fast and stable.

## Benchmark Output

The benchmark report is written to:

```
retrieval_benchmark.md
```

It includes:

- original query
- expanded query
- top-3 results from raw vector search
- top-3 results from expanded-query search
- expected chunk rank, score delta, and verdict
- a short observation for each comparison

The benchmark is intentionally simple. It is not claiming that query expansion is always better. In real systems, query expansion can improve recall, but it can also introduce query drift if the rewritten query adds assumptions that were not present in the user's original intent.

## Error Handling and Validation

The implementation validates common failure cases:

- empty corpus
- invalid corpus JSON
- empty query
- empty vector store search
- embedding dimension mismatch
- missing query expander for Strategy B
- Ollama request failures

These checks are small, but they matter because retrieval systems often fail silently when inputs are malformed.

## Production Migration to Vertex AI Vector Search

For production, I would not keep FAISS inside the application process.

A more production-ready GCP setup would look like this:

```
Documents / Knowledge Sources
   ↓
Chunking Pipeline
   ↓
Vertex AI Embeddings
   ↓
Vertex AI Vector Search Index
   ↓
Retrieval API
   ↓
RAG / Agent Layer
```

The migration path would be:

1. Replace SentenceTransformerEmbeddingProvider with a Vertex AI embedding provider.
2. Export each chunk with:
   - stable chunk ID
   - text
   - metadata
   - embedding vector
3. Build a Vertex AI Vector Search index from those embeddings.
4. Replace FaissVectorStore with a VertexVectorStore implementation.
5. Keep the Retriever interface mostly unchanged.
6. Add production concerns around access control, index refreshes, monitoring, and evaluation.

The important part is that the current code already separates embedding, storage, expansion, and retrieval logic. So migration would mainly involve replacing implementations, not rewriting the whole system.

## Production Considerations

If this were moving beyond an assessment, I would focus on these areas:

### Incremental indexing

The current project loads and indexes the full corpus at startup. That is fine for a small local assessment, but not enough for production.

In production, new or updated documents should be embedded and indexed incrementally.

### Access control

The current corpus has no user-specific access rules.

For a real product, retrieval must enforce authorization before returning chunks. Otherwise, vector search can leak data across users, teams, or tenants.

### Observability

A production retrieval system should track:

- query latency
- embedding latency
- vector search latency
- top-k score distribution
- empty or low-confidence retrievals
- query expansion failures
- expected-chunk hit rate for benchmark queries

These metrics are useful because retrieval issues often show up as quality degradation before they show up as hard failures.

### Query expansion safety

Query expansion can help when the user query is short or vague. But it can also hurt if the model adds unrelated terms.

For production, I would log both the original and expanded query, evaluate whether expansion improves retrieval, and fall back to raw retrieval when expansion confidence is low.

### Evaluation

The current benchmark is a small offline comparison. For production, I would expand this into a retrieval evaluation set with expected relevant chunks for each query.

Useful metrics would include:

- recall@k
- precision@k
- mean reciprocal rank
- expected chunk found or not found
- score gap between rank 1 and lower results

## Limitations

This project intentionally keeps the corpus small because the assessment is focused on retrieval logic and benchmarking.

Current limitations:

- no document chunking pipeline
- no persistent FAISS index on disk
- no reranker
- no answer generation layer
- no real Vertex AI calls
- no user-level authorization
- small benchmark set
- query expansion is deterministic for reproducibility

These are acceptable for the local assessment, but they would need to be addressed before production use.

## Why Ollama Is Included

Ollama is included as a local LLM option for query expansion.

The project does not require Ollama to run tests because tests should not depend on a local model server. The deterministic mock is the default for reliable benchmarking.

If Ollama is running locally, OllamaQueryExpander can be used to rewrite queries through a local model while keeping the same retriever interface.

## Final Notes

The implementation is intentionally small and modular.

I avoided building a full chatbot or adding extra infrastructure because the assignment is specifically about retrieval behavior, embeddings, vector search, query expansion, and benchmarking. The main focus is to make the retrieval comparison easy to run, easy to inspect, and easy to extend.
