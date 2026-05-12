# Retrieval Benchmark: Strategy A vs Strategy B

This benchmark compares two retrieval strategies over the same local technical corpus.

- Strategy A: raw vector search using the original user query.
- Strategy B: AI-enhanced retrieval using query expansion before embedding.

Embeddings are normalized, and FAISS inner-product search is used as cosine-style similarity.

## Query 1

Original query: `How does the system handle peak load?`

Expanded query: `How does the system handle peak load, traffic spikes, queue depth, horizontal worker scaling, stateless API servers, backpressure, and processing latency?`

### Strategy A: Raw Vector Search

|   Rank | Chunk ID   | Title                        |   Score |
|--------|------------|------------------------------|---------|
|      1 | chunk_001  | Peak Load Handling           |  0.6273 |
|      2 | chunk_006  | Database Consistency         |  0.3054 |
|      3 | chunk_004  | Observability and Monitoring |  0.3021 |

### Strategy B: AI-Enhanced Retrieval

|   Rank | Chunk ID   | Title                        |   Score |
|--------|------------|------------------------------|---------|
|      1 | chunk_001  | Peak Load Handling           |  0.833  |
|      2 | chunk_004  | Observability and Monitoring |  0.4614 |
|      3 | chunk_006  | Database Consistency         |  0.4145 |

### Observation

Both strategies retrieved the same top-ranked chunk. The expanded query may still help by changing score distribution or improving recall in lower-ranked results.

## Query 2

Original query: `What happens when background jobs fail repeatedly?`

Expanded query: `What retry strategy is used for repeated background job failures, including exponential backoff, jitter, retry budget, and dead-letter queues?`

### Strategy A: Raw Vector Search

|   Rank | Chunk ID   | Title                        |   Score |
|--------|------------|------------------------------|---------|
|      1 | chunk_003  | Retry and Failure Handling   |  0.6885 |
|      2 | chunk_001  | Peak Load Handling           |  0.2679 |
|      3 | chunk_004  | Observability and Monitoring |  0.2022 |

### Strategy B: AI-Enhanced Retrieval

|   Rank | Chunk ID   | Title                        |   Score |
|--------|------------|------------------------------|---------|
|      1 | chunk_003  | Retry and Failure Handling   |  0.8064 |
|      2 | chunk_001  | Peak Load Handling           |  0.3918 |
|      3 | chunk_004  | Observability and Monitoring |  0.3198 |

### Observation

Both strategies retrieved the same top-ranked chunk. The expanded query may still help by changing score distribution or improving recall in lower-ranked results.

## Query 3

Original query: `How can we know if retrieval quality is getting worse?`

Expanded query: `How is retrieval quality evaluated using top-k results, expected chunk rank, similarity score, benchmark queries, and whether query expansion improves relevance?`

### Strategy A: Raw Vector Search

|   Rank | Chunk ID   | Title                      |   Score |
|--------|------------|----------------------------|---------|
|      1 | chunk_010  | Retrieval Evaluation       |  0.5578 |
|      2 | chunk_003  | Retry and Failure Handling |  0.2414 |
|      3 | chunk_002  | Caching Strategy           |  0.2238 |

### Strategy B: AI-Enhanced Retrieval

|   Rank | Chunk ID   | Title                    |   Score |
|--------|------------|--------------------------|---------|
|      1 | chunk_010  | Retrieval Evaluation     |  0.9333 |
|      2 | chunk_007  | Semantic Search Pipeline |  0.4729 |
|      3 | chunk_008  | Query Expansion          |  0.3592 |

### Observation

Both strategies retrieved the same top-ranked chunk. The expanded query may still help by changing score distribution or improving recall in lower-ranked results.
